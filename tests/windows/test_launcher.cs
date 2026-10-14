using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace NInferWindows {
    internal static class LauncherTests {
        static void Require(bool condition, string message) { if (!condition) throw new Exception(message); }
        static string Value(string[] args, string key) { return args[Array.IndexOf(args, key) + 1]; }
        static void OptionsAndReplies() {
            var options = new LaunchOptions();
            var ordinary = options.Arguments("test model.ninfer", 8081);
            Require(!ordinary.Contains("--spec") && !ordinary.Contains("--no-thinking"), "ordinary/thinking defaults");
            Require(Value(ordinary, "--default-max-tokens") == "2048", "output limit lost");
            Require(Value(ordinary, "--port") == "8081", "selected port lost");
            options.Backend = "dflash"; options.Context = 98304; options.Drafts = 7;
            var dflash = options.Arguments("test model.ninfer", 8081);
            Require(Value(dflash, "--spec") == "dflash" && Value(dflash, "--draft-tokens") == "7", "DFlash selection lost");
            Require(!dflash.Contains("--lm-head-draft") && Value(dflash, "--kv-capacity") == "98304", "DFlash got MTP settings");
            var json = new System.Web.Script.Serialization.JavaScriptSerializer();
            var saved = json.Deserialize<LauncherSettings>(json.Serialize(new LauncherSettings { Model = "test model.ninfer", Options = options }));
            Require(saved.Model == "test model.ninfer" && saved.Options.Arguments(saved.Model, 8081).SequenceEqual(dflash), "saved launch settings changed the command");
            options.Concurrency = 2;
            bool rejected = false; try { options.Arguments("x.ninfer", 8081); } catch { rejected = true; }
            Require(rejected, "known broken DFlash concurrency was admitted");
            options.Backend = "mtp"; options.Drafts = 3; options.Context = 4096; options.Thinking = false;
            options.Graph = false; options.PrefixReuse = false; options.Kv = "bf16";
            var mtp = options.Arguments("test model.ninfer", 8081);
            Require(Value(mtp, "--spec") == "mtp" && mtp.Contains("--lm-head-draft"), "MTP optimized head lost");
            Require(mtp.Contains("--no-thinking") && mtp.Contains("--no-cuda-graph") && mtp.Contains("--no-prefix-reuse"), "disabled options lost");
            Require(Value(mtp, "--kv-capacity") == "8192" && Value(mtp, "--kv-dtype") == "bf16", "shared KV configuration wrong");
            options.OptimizedMtpHead = false;
            Require(!options.Arguments("x.ninfer", 8081).Contains("--lm-head-draft"), "full MTP head ignored");
            string text = Runtime.FormatChat("{\"choices\":[{\"message\":{\"reasoning_content\":\"reason\",\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}");
            Require(text.Contains("reason") && text.Contains("answer"), "thinking or final answer hidden");
            string limited = Runtime.FormatChat("{\"choices\":[{\"message\":{\"reasoning_content\":\"unfinished\",\"content\":null},\"finish_reason\":\"length\"}]}");
            Require(limited.Contains("unfinished") && limited.Contains("输出上限"), "reasoning-only truncation hidden");
            Require(Runtime.FormatChat("{\"choices\":[{\"message\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}") == "hello", "plain reply changed");
            Console.WriteLine("PASS: launch options, DFlash concurrency rejection, reasoning and truncation display");
        }
        static async Task Exercise() {
            string directory = Path.Combine(Path.GetTempPath(), "ninfer-download-" + Guid.NewGuid());
            Directory.CreateDirectory(directory);
            byte[] bytes = new byte[3 * 1024 * 1024 + 127]; new Random(7).NextBytes(bytes);
            string hash;
            using (var sha = SHA256.Create()) hash = BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
            var portProbe = new TcpListener(IPAddress.Loopback, 0); portProbe.Start();
            int port = ((IPEndPoint)portProbe.LocalEndpoint).Port; portProbe.Stop();
            string root = "http://127.0.0.1:" + port + "/";
            using (var server = new HttpListener()) {
                server.Prefixes.Add(root); server.Start();
                long observedResume = -1;
                var serving = Task.Run(async () => {
                    while (server.IsListening) {
                        HttpListenerContext context;
                        try { context = await server.GetContextAsync(); } catch (HttpListenerException) { break; } catch (ObjectDisposedException) { break; }
                        try {
                            long offset = 0;
                            string range = context.Request.Headers["Range"];
                            if (range != null && context.Request.RawUrl != "/ignore") {
                                offset = long.Parse(range.Substring(6).TrimEnd('-')); observedResume = offset;
                                context.Response.StatusCode = 206;
                                context.Response.Headers.Add("Content-Range", "bytes " + offset + "-" + (bytes.Length - 1) + "/" + bytes.Length);
                            }
                            context.Response.ContentLength64 = bytes.Length - offset;
                            await context.Response.OutputStream.WriteAsync(bytes, (int)offset, bytes.Length - (int)offset);
                        } catch (HttpListenerException) { } catch (IOException) { }
                        finally { context.Response.Close(); }
                    }
                });
                try {
                    foreach (string mode in new[] { "resume", "ignore" }) {
                        string target = Path.Combine(directory, mode + ".ninfer");
                        using (var partial = File.Create(target + ".part")) partial.Write(bytes, 0, 12345);
                        await Runtime.DownloadAsync(target, s => { }, CancellationToken.None, root + mode, bytes.Length, hash);
                        Require(File.Exists(target) && new FileInfo(target).Length == bytes.Length, "download length");
                        if (mode == "resume") Require(observedResume == 12345, "did not resume at the saved offset");
                        Require(!File.Exists(target + ".part"), "published file retains partial");
                    }
                    string corrupt = Path.Combine(directory, "corrupt.ninfer");
                    bool rejected = false;
                    try { await Runtime.DownloadAsync(corrupt, s => { }, CancellationToken.None, root + "full", bytes.Length, new string('0', 64)); }
                    catch (InvalidDataException) { rejected = true; }
                    Require(rejected && !File.Exists(corrupt), "bad hash was published");
                    string cancelled = Path.Combine(directory, "cancelled.ninfer");
                    using (var cancellation = new CancellationTokenSource()) {
                        try { await Runtime.DownloadAsync(cancelled, s => cancellation.Cancel(), cancellation.Token, root + "full", bytes.Length, hash); }
                        catch (OperationCanceledException) { }
                        Require(!File.Exists(cancelled) && File.Exists(cancelled + ".part"), "cancel lost progress or published model");
                    }
                    await Runtime.DownloadAsync(cancelled, s => { }, CancellationToken.None, root + "resume", bytes.Length, hash);
                    Require(File.Exists(cancelled), "resume after cancellation failed");
                    Console.WriteLine("PASS: HTTP range resume, range-ignored restart, SHA rejection, cancellation and resume");
                } finally { server.Stop(); }
                await serving;
            }
            // This directory was freshly created above and contains only test fixtures.
            Directory.Delete(directory, true);
        }
        static int Main() {
            try { OptionsAndReplies(); Exercise().GetAwaiter().GetResult(); return 0; }
            catch (Exception error) { Console.Error.WriteLine(error); return 1; }
        }
    }
}
