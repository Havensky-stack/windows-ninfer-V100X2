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
    internal sealed class Runtime : IDisposable {
        internal const string ModelName = "Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer";
        internal const long ModelBytes = 23773675264L;
        internal const string ModelHash = "3c2d39f364a8932c82b41d4c424c940285573b0309481a67c2fde048368bf808";
        internal const string ModelUrl = "https://modelscope.cn/models/Merkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-NVFP4-NInfer/resolve/master/NVFP4-NInfer/W4A4/" + ModelName;
        internal static readonly string StateDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "NInferV100X2");
        internal readonly string Root;
        internal readonly Action<string> Log;
        private Process server;
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

        internal async Task StartAsync(string model, int context, CancellationToken token) {
            if (Running) throw new Exception("服务已经启动。");
            await CheckAsync();
            token.ThrowIfCancellationRequested();
            if (!File.Exists(model) || !string.Equals(Path.GetExtension(model), ".ninfer", StringComparison.OrdinalIgnoreCase)) throw new Exception("请选择有效的 .ninfer 模型文件。");
            var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start(); int port = ((IPEndPoint)listener.LocalEndpoint).Port; listener.Stop();
            Endpoint = "http://127.0.0.1:" + port;
            var args = new[] { Path.GetFullPath(model), "--host", "127.0.0.1", "--port", port.ToString(), "--model-id", "coder390",
                "--tp", "2", "--devices", "0,1", "--max-concurrency", "1", "--max-context", context.ToString(),
                "--kv-capacity", context.ToString(), "--kv-dtype", "int8", "--prefill-chunk", "256", "--default-max-tokens", "1024", "--no-thinking" };
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
            string body = json.Serialize(new { model = "coder390", messages = new[] { new { role = "user", content = text } }, max_tokens = 512, temperature = 0.7 });
            using (var content = new StringContent(body, Encoding.UTF8, "application/json"))
            using (var response = await http.PostAsync(Endpoint + "/v1/chat/completions", content, token)) {
                string answer = await response.Content.ReadAsStringAsync();
                if (!response.IsSuccessStatusCode) throw new Exception("请求失败：" + answer);
                var parsed = json.Deserialize<Dictionary<string, object>>(answer);
                var choice = (Dictionary<string, object>)((System.Collections.IList)parsed["choices"])[0];
                return (string)((Dictionary<string, object>)choice["message"])["content"];
            }
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
        readonly ComboBox context = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 210 };
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
        readonly string settings = Path.Combine(Runtime.StateDir, "model-path.txt");
        readonly object logLock = new object();
        bool busy;

        internal Launcher(bool preview = false) {
            Directory.CreateDirectory(Runtime.StateDir);
            runtime = new Runtime(AppDomain.CurrentDomain.BaseDirectory, AddLog);
            Text = "NInfer · Windows 双 V100 · 0.1.0 预览版";
            Font = new Font("Microsoft YaHei UI", 10); Size = new Size(1000, 800); MinimumSize = new Size(850, 680);
            StartPosition = FormStartPosition.CenterScreen;
            var layout = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(20), ColumnCount = 1, RowCount = 9 };
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 45));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 90));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 45));
            Controls.Add(layout);
            layout.Controls.Add(new Label { Text = "本地运行 Qwen · 不改驱动 · 无需开发环境", Font = new Font(Font.FontFamily, 17, FontStyle.Bold), AutoSize = true });
            layout.Controls.Add(new Label { Text = "面向 2 × V100 16GB，模型约占 23.8GB 磁盘。默认普通解码，MTP 尚未通过严格校验。\r\n128K 为建议容量；20万仅验证过短请求。游戏或其他模型会占用显存。", AutoSize = true });
            var modelRow = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2 };
            modelRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100)); modelRow.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            modelRow.Controls.Add(model); modelRow.Controls.Add(browse); layout.Controls.Add(modelRow);
            var actions = new FlowLayoutPanel { Dock = DockStyle.Fill };
            context.Items.AddRange(new object[] { "16K · 显存占用较低", "64K", "128K · 推荐", "200000 · 余量较小" }); context.SelectedIndex = 2;
            actions.Controls.Add(context); actions.Controls.Add(start); actions.Controls.Add(stop); actions.Controls.Add(download); layout.Controls.Add(actions);
            layout.Controls.Add(status);
            var api = new FlowLayoutPanel { Dock = DockStyle.Fill };
            var copy = new Button { Text = "复制 API 地址", AutoSize = true };
            copy.Click += (s, e) => { if (runtime.Endpoint != null) Clipboard.SetText(runtime.Endpoint + "/v1"); };
            api.Controls.Add(address); api.Controls.Add(copy); api.Controls.Add(send); layout.Controls.Add(api);
            prompt.Text = "用中文简单介绍一下你能帮助我做什么。";
            layout.Controls.Add(Section("输入问题", prompt)); layout.Controls.Add(Section("回答（每次独立问答）", answer)); layout.Controls.Add(Section("运行日志", logs));
            if (File.Exists(settings)) model.Text = File.ReadAllText(settings, Encoding.UTF8);
            browse.Click += (s, e) => {
                using (var dialog = new OpenFileDialog { Filter = "NInfer 模型|*.ninfer", Title = "选择本项目登记的 .ninfer 模型" })
                    if (dialog.ShowDialog(this) == DialogResult.OK) { model.Text = dialog.FileName; SaveModel(); }
            };
            start.Click += async (s, e) => await Work(async token => {
                status.Text = "正在加载模型，请稍候…"; SaveModel();
                await runtime.StartAsync(model.Text, new[] { 16384, 65536, 131072, 200000 }[context.SelectedIndex], token);
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
                        model.Text = target; SaveModel(); status.Text = "模型已准备好，点击“启动模型”";
                    });
                }
            };
            if (!preview) Shown += async (s, e) => await Work(async token => { AddLog(await runtime.CheckAsync()); status.Text = "环境就绪 · 选择模型后启动，或下载推荐模型"; });
            FormClosing += (s, e) => { operation.Cancel(); runtime.Stop(); };
            FormClosed += (s, e) => runtime.Dispose();
            UpdateButtons();
        }
        static GroupBox Section(string title, Control content) {
            var box = new GroupBox { Text = title, Dock = DockStyle.Fill, Padding = new Padding(8) };
            box.Controls.Add(content); return box;
        }
        void SaveModel() { File.WriteAllText(settings, model.Text, Encoding.UTF8); }
        void UpdateButtons() {
            if (IsDisposed) return;
            start.Enabled = !busy && !runtime.Running; browse.Enabled = !busy && !runtime.Running;
            download.Enabled = !busy && !runtime.Running; context.Enabled = !busy && !runtime.Running;
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
                if (args.Length == 3 && args[0] == "--smoke") {
                    using (var runtime = new Runtime(AppDomain.CurrentDomain.BaseDirectory, line => File.AppendAllText(args[2], line + "\n"))) {
                        runtime.StartAsync(args[1], 4096, CancellationToken.None).GetAwaiter().GetResult();
                        var answer = runtime.ChatAsync("Reply in Chinese with one short sentence: what is 2 plus 2?", CancellationToken.None).GetAwaiter().GetResult();
                        if (string.IsNullOrWhiteSpace(answer)) throw new Exception("No answer returned");
                        File.AppendAllText(args[2], "ANSWER: " + answer + "\nSMOKE PASS\n");
                    }
                    return 0;
                }
                Application.Run(new Launcher()); return 0;
            } catch (Exception error) {
                if (args.Length == 3 && args[0] == "--smoke") File.AppendAllText(args[2], error.ToString());
                else MessageBox.Show(error.Message, "NInfer 启动失败");
                return 1;
            }
        }
    }
}
