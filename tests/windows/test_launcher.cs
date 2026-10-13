using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace NInferWindows {
    internal static class LauncherTests {
        static void Require(bool condition, string message) { if (!condition) throw new Exception(message); }
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
            try { Exercise().GetAwaiter().GetResult(); return 0; }
            catch (Exception error) { Console.Error.WriteLine(error); return 1; }
        }
    }
}
