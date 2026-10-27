using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

namespace NInferWindows {
    internal sealed class LaunchOptions {
        public string Backend = "none";
        public int Drafts = 3;
        public int Context = 131072;
        public int OutputTokens = 2048;
        public int PrefillChunk = 256;
        public int Port = 0;
        public int Concurrency = 1;
        public string Kv = "int8";
        public bool Thinking = true;
        public bool Graph = true;
        public bool PrefixReuse = true;
        public bool OptimizedMtpHead = true;

        internal void Validate() {
            if (Backend != "none" && Backend != "mtp" && Backend != "dflash") throw new Exception("请选择普通、MTP 或 DFlash 解码。");
            if (Context < 1024 || Context > 262144) throw new Exception("上下文容量须为 1024–262144；实际可用值取决于显存。");
            if (OutputTokens < 16 || OutputTokens > Context) throw new Exception("输出上限须为 16 到上下文容量之间，包含思考与回答。");
            if (PrefillChunk < 128 || PrefillChunk > 1024 || PrefillChunk % 128 != 0) throw new Exception("预填充分块须为 128–1024 之间的 128 倍数。");
            if (Port < 0 || Port > 65535) throw new Exception("端口须为 0（自动）或 1–65535。");
            if (Concurrency < 1 || Concurrency > 8) throw new Exception("并发数须为 1–8。");
            if (Backend == "dflash" && Concurrency != 1) throw new Exception("DFlash 多并发尚有已知错误，请使用单请求。");
            if (Backend != "none" && (Drafts < 1 || Drafts > (Backend == "mtp" ? 3 : 7))) throw new Exception("MTP 草稿数为 1–3；DFlash 为 1–7。");
            if (Kv != "int8" && Kv != "bf16") throw new Exception("KV 格式须为 int8 或 bf16。");
        }

        internal string[] Arguments(string model, int port) {
            Validate();
            var args = new List<string> { Path.GetFullPath(model), "--host", "127.0.0.1", "--port", port.ToString(), "--model-id", "coder390",
                "--tp", "2", "--devices", "0,1", "--max-concurrency", Concurrency.ToString(), "--max-context", Context.ToString(),
                "--kv-capacity", ((long)Context * Concurrency).ToString(), "--kv-dtype", Kv, "--prefill-chunk", PrefillChunk.ToString(),
                "--default-max-tokens", OutputTokens.ToString() };
            if (!Thinking) args.Add("--no-thinking");
            if (!Graph) args.Add("--no-cuda-graph");
            if (!PrefixReuse) args.Add("--no-prefix-reuse");
            if (Backend != "none") { args.AddRange(new[] { "--spec", Backend, "--draft-tokens", Drafts.ToString() }); }
            if (Backend == "mtp" && OptimizedMtpHead) args.Add("--lm-head-draft");
            return args.ToArray();
        }
    }

    internal sealed class LauncherSettings {
        public string Model = "";
        public LaunchOptions Options = new LaunchOptions();
    }

    internal sealed class Runtime : IDisposable {
        internal const string ModelName = "Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer";
        internal const long ModelBytes = 23773675264L;
        internal const string ModelHash = "3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808";
        internal const string ModelUrl = "https://modelscope.cn/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer/resolve/master/NVFP4-NInfer/W4A4/" + ModelName;
        internal static readonly string StateDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "NInferV100X2");
        internal readonly string Root;
        internal readonly Action<string> Log;
        private Process server;
        private LaunchOptions activeOptions;
        private readonly HttpClient http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        internal string Endpoint { get; private set; }
        internal bool Running { get { return server != null && !server.HasExited; } }

        internal Runtime(string root, Action<string> log) { Root = root; Log = log; }

        // CommandLineToArgvW/CRT quoting, including paths ending in a backslash.
        internal static string Quote(string value) {
            var result = new StringBuilder("\"");
            int slashes = 0;
            foreach (char c in value) {
                if (c == '\\') { slashes++; continue; }
                result.Append('\\', c == '"' ? slashes * 2 + 1 : slashes);
                result.Append(c); slashes = 0;
            }
            return result.Append('\\', slashes * 2).Append('"').ToString();
        }

        internal async Task<string> CheckAsync() {
            return await Task.Run(() => {
                if (!Environment.Is64BitOperatingSystem || Environment.OSVersion.Version.Major < 10)
                    throw new Exception("需要 Windows 10/11 64 位；本版本实测 Windows 11。");
                string smi = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "nvidia-smi.exe");
                if (!File.Exists(smi)) smi = "nvidia-smi.exe";
                var info = new ProcessStartInfo(smi, "--query-gpu=index,name,memory.total,driver_version,driver_model.current --format=csv,noheader,nounits") {
                    UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true
                };
                using (var process = Process.Start(info)) {
                    var outputTask = process.StandardOutput.ReadToEndAsync();
                    var errorTask = process.StandardError.ReadToEndAsync();
                    if (!process.WaitForExit(15000)) { process.Kill(); throw new Exception("显卡检测超时，请检查 NVIDIA 驱动。"); }
                    string output = outputTask.GetAwaiter().GetResult();
                    if (process.ExitCode != 0) throw new Exception("NVIDIA 驱动检测失败：" + errorTask.GetAwaiter().GetResult());
                    var rows = output.Trim().Split('\n').Select(line => line.Trim().Split(',').Select(s => s.Trim()).ToArray()).ToArray();
                    foreach (string index in new[] { "0", "1" }) {
                        var row = rows.FirstOrDefault(r => r.Length >= 5 && r[0] == index);
                        int memory;
                        Version driver;
                        if (row == null || !row[1].Contains("V100") || !int.TryParse(row[2], out memory) || memory < 15000)
                            throw new Exception("此预览版需要 NVIDIA 设备 0、1 均为至少 16GB 的 V100。\r\n" + output);
                        if (!Version.TryParse(row[3], out driver) || driver < new Version(528, 33))
                            throw new Exception("CUDA 12.x 需要至少 528.33 的 Windows 驱动；本项目实测 GRID 553.62。请自行安装合适驱动。");
                    }
                    string exe = Path.Combine(Root, "bin", "ninfer-serve.exe");
                    if (!File.Exists(exe)) throw new Exception("缺少 bin\\ninfer-serve.exe，请完整解压发布包，不要单独复制启动器。");
                    return output.Trim() + "\r\n环境检查通过。无需安装 Python、CUDA Toolkit 或 Visual Studio。";
                }
            });
        }

        internal static async Task DownloadAsync(string target, Action<string> progress, CancellationToken token, string url = ModelUrl, long expectedBytes = ModelBytes, string expectedHash = ModelHash) {
            string partial = target + ".part";
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(target)));
            if (File.Exists(target)) throw new Exception("目标模型已存在，请直接选择该文件；不会覆盖现有模型。");
            long present = File.Exists(partial) ? new FileInfo(partial).Length : 0;
            if (present > expectedBytes) throw new Exception("下载缓存大小异常，请移走该 .part 文件后重试。");
            var drive = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(target)));
            if (drive.AvailableFreeSpace < expectedBytes - present + 1024L * 1024 * 1024)
                throw new Exception("磁盘空间不足，请选择至少有 25GB 可用空间的位置。");
            using (var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan }) {
                for (int attempt = 0; present < expectedBytes; attempt++) {
                    token.ThrowIfCancellationRequested();
                    try {
                        using (var request = new HttpRequestMessage(HttpMethod.Get, url)) {
                            if (present > 0) request.Headers.Range = new RangeHeaderValue(present, null);
                            using (var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, token)) {
                                response.EnsureSuccessStatusCode();
                                var range = response.Content.Headers.ContentRange;
                                if (response.StatusCode == HttpStatusCode.PartialContent) {
                                    if (range == null || range.From != present || range.Length != expectedBytes)
                                        throw new InvalidDataException("下载服务器返回了不匹配的文件范围。");
                                } else if (response.StatusCode == HttpStatusCode.OK) { present = 0; }
                                else throw new InvalidDataException("下载服务器返回了不支持的响应。");
                                using (var source = await response.Content.ReadAsStreamAsync())
                                using (var destination = new FileStream(partial, FileMode.OpenOrCreate, FileAccess.Write, FileShare.Read, 1048576, true)) {
                                    destination.SetLength(present); destination.Position = present;
                                    byte[] buffer = new byte[1048576];
                                    var clock = Stopwatch.StartNew();
                                    long last = -1000;
                                    int count;
                                    while ((count = await source.ReadAsync(buffer, 0, buffer.Length, token)) != 0) {
                                        if (present + count > expectedBytes) throw new InvalidDataException("下载内容超过登记的模型大小。");
                                        await destination.WriteAsync(buffer, 0, count, token); present += count;
                                        if (clock.ElapsedMilliseconds - last > 500) {
                                            progress(string.Format("下载 {0:F1}% · {1:F2} / {2:F2} GB", 100.0 * present / expectedBytes, present / 1e9, expectedBytes / 1e9));
                                            last = clock.ElapsedMilliseconds;
                                        }
                                    }
                                }
                                if (present != expectedBytes) throw new IOException("连接提前结束。");
                            }
                        }
                    } catch (Exception error) {
                        if (token.IsCancellationRequested || error is InvalidDataException || attempt >= 7) throw;
                        present = File.Exists(partial) ? new FileInfo(partial).Length : 0;
                        progress("连接中断，保留进度并重试：" + error.Message);
                    }
                    if (present < expectedBytes) await Task.Delay(2000, token);
                }
            }
            progress("下载完成，正在校验 SHA-256（可能需要一两分钟）…");
            string hash = await Task.Run(() => {
                using (var sha = SHA256.Create())
                using (var input = File.OpenRead(partial)) {
                    byte[] buffer = new byte[4 * 1024 * 1024]; int count;
                    while ((count = input.Read(buffer, 0, buffer.Length)) != 0) {
                        token.ThrowIfCancellationRequested(); sha.TransformBlock(buffer, 0, count, buffer, 0);
                    }
                    sha.TransformFinalBlock(new byte[0], 0, 0);
                    return BitConverter.ToString(sha.Hash).Replace("-", "").ToLowerInvariant();
                }
            }, token);
            if (hash != expectedHash) throw new InvalidDataException("模型 SHA-256 不匹配；保留 .part 文件供排查，请移走该文件后重新下载。");
            token.ThrowIfCancellationRequested();
            File.Move(partial, target);
            progress("模型下载与完整性校验通过。");
        }

        internal async Task StartAsync(string model, LaunchOptions options, CancellationToken token) {
            if (Running) throw new Exception("服务已经启动。");
            options.Validate();
            await CheckAsync();
            token.ThrowIfCancellationRequested();
            if (!File.Exists(model) || !string.Equals(Path.GetExtension(model), ".ninfer", StringComparison.OrdinalIgnoreCase)) throw new Exception("请选择有效的 .ninfer 模型文件。");
            int port = options.Port;
            if (port == 0) {
                var listener = new TcpListener(IPAddress.Loopback, 0);
                listener.Start(); port = ((IPEndPoint)listener.LocalEndpoint).Port; listener.Stop();
            } else {
                var listener = new TcpListener(IPAddress.Loopback, port);
                try { listener.Start(); } catch (SocketException) { throw new Exception("端口已被占用，请换一个端口或设为 0 自动选择。"); }
                finally { listener.Stop(); }
            }
            Endpoint = "http://127.0.0.1:" + port;
            var args = options.Arguments(model, port);
            activeOptions = options;
            Log("启动命令：ninfer-serve.exe " + string.Join(" ", args.Select(Quote)));
            server = new Process { StartInfo = new ProcessStartInfo(Path.Combine(Root, "bin", "ninfer-serve.exe"), string.Join(" ", args.Select(Quote))) {
                WorkingDirectory = Path.Combine(Root, "bin"), UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardOutput = true, RedirectStandardError = true, StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
            }};
            server.OutputDataReceived += (s, e) => { if (e.Data != null) Log(e.Data); };
            server.ErrorDataReceived += (s, e) => { if (e.Data != null) Log(e.Data); };
            try {
                server.Start(); server.BeginOutputReadLine(); server.BeginErrorReadLine();
                for (int i = 0; i < 240; i++) {
                    token.ThrowIfCancellationRequested();
                    if (!Running) throw new Exception("推理服务启动失败，请查看日志。显存不足时降低上下文，并自行关闭其他占用显存的模型。");
                    using (var probe = CancellationTokenSource.CreateLinkedTokenSource(token)) {
                        probe.CancelAfter(1000);
                        try {
                            using (var response = await http.GetAsync(Endpoint + "/health", probe.Token)) {
                                if (response.IsSuccessStatusCode) { Log("服务已就绪：" + Endpoint + "/v1"); return; }
                            }
                        } catch (HttpRequestException) { } catch (TaskCanceledException) { token.ThrowIfCancellationRequested(); }
                    }
                    await Task.Delay(1000, token);
                }
                throw new Exception("加载模型超时，请检查日志。");
            } catch { Stop(); throw; }
        }

        internal async Task<string> ChatAsync(string text, CancellationToken token) {
            if (!Running) throw new Exception("请先启动模型。");
            var json = new JavaScriptSerializer();
            string body = json.Serialize(new { model = "coder390", messages = new[] { new { role = "user", content = text } }, max_tokens = activeOptions.OutputTokens });
            using (var content = new StringContent(body, Encoding.UTF8, "application/json"))
            using (var response = await http.PostAsync(Endpoint + "/v1/chat/completions", content, token)) {
                string answer = await response.Content.ReadAsStringAsync();
                if (!response.IsSuccessStatusCode) throw new Exception("请求失败：" + answer);
                return FormatChat(answer);
            }
        }
        internal static string FormatChat(string response) {
            var parsed = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(response);
            var choice = (Dictionary<string, object>)((System.Collections.IList)parsed["choices"])[0];
            var message = (Dictionary<string, object>)choice["message"];
            object reasoning, content, finish;
            message.TryGetValue("reasoning_content", out reasoning); message.TryGetValue("content", out content);
            choice.TryGetValue("finish_reason", out finish);
            string text = string.IsNullOrEmpty(reasoning as string) ? "" : "【思考】\r\n" + reasoning + "\r\n\r\n【回答】\r\n";
            text += content as string ?? "";
            if (string.Equals(finish as string, "length", StringComparison.Ordinal)) text += "\r\n\r\n[已达到输出上限，思考与回答可能未完成；可停止服务后提高输出上限。]";
            return text;
        }
        internal void Stop() {
            if (server == null) return;
            try { if (!server.HasExited) { server.Kill(); server.WaitForExit(5000); } }
            catch (InvalidOperationException) { }
            finally { server.Dispose(); server = null; Endpoint = null; }
        }
        public void Dispose() { Stop(); http.Dispose(); }
    }

    internal sealed class Launcher : Form {
        readonly TextBox model = new TextBox { Dock = DockStyle.Fill };
        readonly NumericUpDown context = Number(1024, 262144, 131072, 108, 1024);
        readonly NumericUpDown outputTokens = Number(16, 262144, 2048, 90, 256);
        readonly NumericUpDown prefill = Number(128, 1024, 256, 72, 128);
        readonly NumericUpDown port = Number(0, 65535, 0, 78, 1);
        readonly NumericUpDown concurrency = Number(1, 8, 1, 50, 1);
        readonly NumericUpDown drafts = Number(1, 3, 3, 50, 1);
        readonly ComboBox backend = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 155 };
        readonly ComboBox kv = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 82 };
        readonly CheckBox thinking = new CheckBox { Text = "开启思考", Checked = true, AutoSize = true };
        readonly CheckBox graph = new CheckBox { Text = "CUDA Graph", Checked = true, AutoSize = true };
        readonly CheckBox prefix = new CheckBox { Text = "前缀复用", Checked = true, AutoSize = true };
        readonly CheckBox mtpHead = new CheckBox { Text = "MTP 优化草稿头", Checked = true, AutoSize = true };
        readonly TableLayoutPanel optionsPanel = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 3, ColumnCount = 1 };
        readonly Button tcc = new Button { Text = "切换 TCC", AutoSize = true };
        readonly Button wddm = new Button { Text = "恢复 WDDM", AutoSize = true };
        readonly TextBox prompt = new TextBox { Multiline = true, Dock = DockStyle.Fill, ScrollBars = ScrollBars.Vertical };
        readonly TextBox answer = new TextBox { Multiline = true, ReadOnly = true, Dock = DockStyle.Fill, ScrollBars = ScrollBars.Vertical };
        readonly TextBox logs = new TextBox { Multiline = true, ReadOnly = true, Dock = DockStyle.Fill, ScrollBars = ScrollBars.Vertical };
        readonly Label status = new Label { AutoSize = true, Text = "正在检查环境…", Padding = new Padding(0, 8, 0, 8) };
        readonly TextBox address = new TextBox { ReadOnly = true, Width = 330, Text = "服务启动后显示 API 地址" };
        readonly Button start = new Button { Text = "启动模型", AutoSize = true };
        readonly Button stop = new Button { Text = "停止 / 取消", AutoSize = true };
        readonly Button send = new Button { Text = "发送单轮提问", AutoSize = true };
        readonly Button download = new Button { Text = "下载推荐模型（23.8 GB）", AutoSize = true };
        readonly Button browse = new Button { Text = "选择本地模型…", AutoSize = true };
        readonly Runtime runtime;
        CancellationTokenSource operation = new CancellationTokenSource();
        readonly string settings = Path.Combine(Runtime.StateDir, "settings.json");
        readonly object logLock = new object();
        bool busy;

        internal Launcher(bool preview = false) {
            Directory.CreateDirectory(Runtime.StateDir);
            runtime = new Runtime(AppDomain.CurrentDomain.BaseDirectory, AddLog);
            var appVersion = System.Reflection.Assembly.GetExecutingAssembly().GetName().Version;
            Text = "NInfer · Windows 双 V100 · " +
                (appVersion.Major == 0 && appVersion.Minor == 0 ? "源码构建" : appVersion.ToString(3) + " 预览版");
            Font = new Font("Microsoft YaHei UI", 10); Size = new Size(1100, 960); MinimumSize = new Size(1060, 900);
            StartPosition = FormStartPosition.CenterScreen;
            var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(20), ColumnCount = 1, RowCount = 10 };
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 174));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 45));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 90));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 45));
            Controls.Add(layout);
            layout.Controls.Add(new Label { Text = "本地运行 Qwen · 双 V100 · 无需开发环境", Font = new Font(Font.FontFamily, 17, FontStyle.Bold), AutoSize = true });
            layout.Controls.Add(new Label { Text = "默认普通解码与思考开启；MTP / DFlash 为实验功能，严格校验仍有 1/64 差异。\r\nDFlash 建议 96K / 单请求；容量不代表占用。WDDM 小消息优化自动启用，TCC 不保证 P2P。", AutoSize = true });
            var modelRow = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2 };
            modelRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100)); modelRow.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            modelRow.Controls.Add(model); modelRow.Controls.Add(browse); layout.Controls.Add(modelRow);
            backend.Items.AddRange(new object[] { "普通解码（默认）", "MTP（实验）", "DFlash（实验）" }); backend.SelectedIndex = 0;
            kv.Items.AddRange(new object[] { "INT8", "BF16" }); kv.SelectedIndex = 0;
            optionsPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));
            optionsPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));
            optionsPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));
            optionsPanel.Controls.Add(OptionRow("解码", backend, "草稿数", drafts, thinking, mtpHead));
            optionsPanel.Controls.Add(OptionRow("上下文容量", context, "输出上限", outputTokens, "预填充分块", prefill, "端口（0=自动）", port));
            optionsPanel.Controls.Add(OptionRow("KV 格式", kv, "并发", concurrency, graph, prefix));
            layout.Controls.Add(Section("模型启动选项（停止服务后修改；输出上限包含思考与回答）", optionsPanel));
            var actions = new FlowLayoutPanel { Dock = DockStyle.Fill };
            actions.Controls.Add(start); actions.Controls.Add(stop); actions.Controls.Add(download); actions.Controls.Add(tcc); actions.Controls.Add(wddm); layout.Controls.Add(actions);
            layout.Controls.Add(status);
            var api = new FlowLayoutPanel { Dock = DockStyle.Fill };
            var copy = new Button { Text = "复制 API 地址", AutoSize = true };
            copy.Click += (s, e) => { if (runtime.Endpoint != null) Clipboard.SetText(runtime.Endpoint + "/v1"); };
            api.Controls.Add(address); api.Controls.Add(copy); api.Controls.Add(send); layout.Controls.Add(api);
            prompt.Text = "用中文简单介绍一下你能帮助我做什么。";
            layout.Controls.Add(Section("输入问题", prompt)); layout.Controls.Add(Section("回答（每次独立问答）", answer)); layout.Controls.Add(Section("运行日志", logs));
            backend.SelectedIndexChanged += (s, e) => {
                drafts.Maximum = backend.SelectedIndex == 2 ? 7 : 3;
                drafts.Value = drafts.Maximum;
                if (backend.SelectedIndex == 2) { context.Value = Math.Min(context.Value, 98304); concurrency.Value = 1; }
                UpdateButtons();
            };
            if (!preview) LoadSettings();
            browse.Click += (s, e) => {
                using (var dialog = new OpenFileDialog { Filter = "NInfer 模型|*.ninfer", Title = "选择本项目登记的 .ninfer 模型" })
                    if (dialog.ShowDialog(this) == DialogResult.OK) { model.Text = dialog.FileName; SaveSettings(); }
            };
            start.Click += async (s, e) => await Work(async token => {
                var options = ReadOptions(); options.Validate();
                status.Text = "正在加载模型，请稍候…"; SaveSettings();
                await runtime.StartAsync(model.Text, options, token);
                address.Text = runtime.Endpoint + "/v1"; status.Text = "运行中 · 可直接提问，或连接支持 OpenAI API 的客户端（模型名 coder390）";
            });
            stop.Click += (s, e) => { operation.Cancel(); runtime.Stop(); address.Text = "服务已停止"; status.Text = "已停止；下载进度保存在 .part 文件中"; UpdateButtons(); };
            send.Click += async (s, e) => await Work(async token => {
                if (string.IsNullOrWhiteSpace(prompt.Text)) throw new Exception("请先输入问题。");
                status.Text = "正在生成（单轮问答，不自动携带历史）…";
                answer.Text = await runtime.ChatAsync(prompt.Text, token); status.Text = "回答完成 · 可继续提问";
            });
            download.Click += async (s, e) => {
                using (var dialog = new FolderBrowserDialog { Description = "选择模型保存目录（需至少25GB可用空间；支持断点续传）" }) {
                    if (dialog.ShowDialog(this) != DialogResult.OK) return;
                    string target = Path.Combine(dialog.SelectedPath, Runtime.ModelName);
                    await Work(async token => {
                        await Runtime.DownloadAsync(target, message => { status.Text = message; }, token);
                        model.Text = target; SaveSettings(); status.Text = "模型已准备好，点击“启动模型”";
                    });
                }
            };
            tcc.Click += (s, e) => SwitchMode("enable-tcc.cmd");
            wddm.Click += (s, e) => SwitchMode("restore-wddm.cmd");
            if (!preview) Shown += async (s, e) => await Work(async token => { AddLog(await runtime.CheckAsync()); status.Text = "环境就绪 · 选择模型后启动，或下载推荐模型"; });
            else status.Text = "选择本地模型，或点击“下载推荐模型”；启动前可调整以上选项。";
            FormClosing += (s, e) => { if (!preview) SaveSettings(); operation.Cancel(); runtime.Stop(); };
            FormClosed += (s, e) => runtime.Dispose();
            UpdateButtons();
        }
        static GroupBox Section(string title, Control content) {
            var box = new GroupBox { Text = title, Dock = DockStyle.Fill, Padding = new Padding(8) };
            box.Controls.Add(content); return box;
        }
        static NumericUpDown Number(int min, int max, int value, int width, int step) {
            return new NumericUpDown { Minimum = min, Maximum = max, Value = value, Width = width, Increment = step, ThousandsSeparator = true };
        }
        static FlowLayoutPanel OptionRow(params object[] items) {
            var row = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
            foreach (object item in items) row.Controls.Add(item is string ? new Label { Text = (string)item, AutoSize = true, Padding = new Padding(0, 5, 0, 0) } : (Control)item);
            return row;
        }
        LaunchOptions ReadOptions() {
            return new LaunchOptions { Backend = new[] { "none", "mtp", "dflash" }[backend.SelectedIndex], Drafts = (int)drafts.Value,
                Context = (int)context.Value, OutputTokens = (int)outputTokens.Value, PrefillChunk = (int)prefill.Value, Port = (int)port.Value,
                Concurrency = (int)concurrency.Value, Kv = kv.SelectedIndex == 0 ? "int8" : "bf16", Thinking = thinking.Checked,
                Graph = graph.Checked, PrefixReuse = prefix.Checked, OptimizedMtpHead = mtpHead.Checked };
        }
        void SaveSettings() {
            try { File.WriteAllText(settings, new JavaScriptSerializer().Serialize(new LauncherSettings { Model = model.Text, Options = ReadOptions() }), Encoding.UTF8); }
            catch (Exception error) { AddLog("设置保存失败：" + error.Message); }
        }
        void LoadSettings() {
            if (!File.Exists(settings)) return;
            try {
                var saved = new JavaScriptSerializer().Deserialize<LauncherSettings>(File.ReadAllText(settings, Encoding.UTF8));
                var o = saved.Options; o.Validate(); model.Text = saved.Model;
                backend.SelectedIndex = Array.IndexOf(new[] { "none", "mtp", "dflash" }, o.Backend);
                drafts.Value = o.Drafts; context.Value = o.Context; outputTokens.Value = o.OutputTokens; prefill.Value = o.PrefillChunk;
                port.Value = o.Port; concurrency.Value = o.Concurrency; kv.SelectedIndex = o.Kv == "int8" ? 0 : 1;
                thinking.Checked = o.Thinking; graph.Checked = o.Graph; prefix.Checked = o.PrefixReuse; mtpHead.Checked = o.OptimizedMtpHead;
            } catch (Exception error) { AddLog("无法恢复保存的设置，请重新选择：" + error.Message); }
        }
        void SwitchMode(string name) {
            try {
                string path = Path.Combine(runtime.Root, "tools", name);
                if (!File.Exists(path)) throw new Exception("缺少模式切换脚本，请完整解压发布包。");
                Process.Start(new ProcessStartInfo(path) { UseShellExecute = true, WorkingDirectory = Path.GetDirectoryName(path) });
                status.Text = "请在脚本窗口完成操作；确认待生效模式后保存工作并重启。";
            } catch (Exception error) { MessageBox.Show(this, error.Message, "显卡模式"); }
        }
        void UpdateButtons() {
            if (IsDisposed) return;
            start.Enabled = !busy && !runtime.Running; browse.Enabled = !busy && !runtime.Running;
            download.Enabled = !busy && !runtime.Running; optionsPanel.Enabled = !busy && !runtime.Running;
            drafts.Enabled = backend.SelectedIndex != 0; mtpHead.Enabled = backend.SelectedIndex == 1;
            concurrency.Enabled = backend.SelectedIndex != 2;
            tcc.Enabled = wddm.Enabled = !busy && !runtime.Running;
            model.ReadOnly = busy || runtime.Running; send.Enabled = !busy && runtime.Running;
        }
        async Task Work(Func<CancellationToken, Task> action) {
            if (busy) return;
            busy = true; operation.Dispose(); operation = new CancellationTokenSource(); UpdateButtons();
            try { await action(operation.Token); }
            catch (OperationCanceledException) { if (!IsDisposed) status.Text = "已取消"; }
            catch (Exception error) { AddLog(error.ToString()); if (!IsDisposed) { status.Text = "操作失败，请查看下方日志"; MessageBox.Show(this, error.Message, "NInfer", MessageBoxButtons.OK, MessageBoxIcon.Information); } }
            finally { busy = false; if (!IsDisposed) UpdateButtons(); }
        }
        void AddLog(string message) {
            lock (logLock) { File.AppendAllText(Path.Combine(Runtime.StateDir, "launcher.log"), message + Environment.NewLine, Encoding.UTF8); }
            if (IsDisposed || !IsHandleCreated) return;
            try { BeginInvoke((Action)(() => { if (logs.TextLength > 100000) logs.Clear(); logs.AppendText(message + Environment.NewLine); })); }
            catch (InvalidOperationException) { }
        }
    }

    internal static class Program {
        [STAThread]
        static int Main(string[] args) {
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
            try {
                if (args.Length == 2 && args[0] == "--render-ui") {
                    using (var form = new Launcher(true)) {
                        form.StartPosition = FormStartPosition.Manual; form.Location = new Point(-32000, -32000); form.ShowInTaskbar = false;
                        form.Show(); Application.DoEvents();
                        using (var bitmap = new Bitmap(form.Width, form.Height)) { form.DrawToBitmap(bitmap, new Rectangle(0, 0, form.Width, form.Height)); bitmap.Save(args[1]); }
                    }
                    return 0;
                }
                if (args.Length == 4 && args[0] == "--smoke") {
                    using (var runtime = new Runtime(AppDomain.CurrentDomain.BaseDirectory, line => File.AppendAllText(args[2], line + "\n"))) {
                        var options = new LaunchOptions { Backend = args[3], Context = 4096, OutputTokens = 512, Drafts = args[3] == "dflash" ? 7 : 3 };
                        runtime.StartAsync(args[1], options, CancellationToken.None).GetAwaiter().GetResult();
                        var answer = runtime.ChatAsync("Reply in Chinese with one short sentence: what is 2 plus 2?", CancellationToken.None).GetAwaiter().GetResult();
                        if (string.IsNullOrWhiteSpace(answer)) throw new Exception("No answer returned");
                        File.AppendAllText(args[2], "ANSWER: " + answer + "\nSMOKE PASS\n");
                    }
                    return 0;
                }
                Application.Run(new Launcher()); return 0;
            } catch (Exception error) {
                if (args.Length == 4 && args[0] == "--smoke") File.AppendAllText(args[2], error.ToString());
                else MessageBox.Show(error.Message, "NInfer 启动失败");
                return 1;
            }
        }
    }
}
