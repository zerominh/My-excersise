using CommandLine;
using Google.Cloud.Speech.V1;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace GoogleCloudSamples
{
    //class ListenOptions
    //{
    //    public int Seconds { get; set; } = 10;
    //}
    public class Recognize
    {
        public static string program = "idea64";
        public static string fileName = "data.txt";
        public static string[] mapTable;
        public static string s = "";

        [DllImport("User32.dll")]
        static extern int SetForegroundWindow(IntPtr point);
        [DllImport("user32.dll")]
        public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);
        static void SendKey(string s)
        {
            Process proc = Process.GetProcessesByName(program).FirstOrDefault();

            if (proc != null && proc.MainWindowHandle != IntPtr.Zero)
            {
                SetForegroundWindow(proc.MainWindowHandle);
                SendKeys.SendWait(s);
            }
            else
            {
                proc.Kill();
                Console.WriteLine("can't found process" + program);
            }
        }

        static void SetMapTable()
        {
            const Int32 BufferSize = 128;
            using (var fileStream = File.OpenRead(fileName))
            using (var streamReader = new StreamReader(fileStream, Encoding.UTF8, true, BufferSize))
            {
                List<string> list = new List<string>();
                string line;
                while ((line = streamReader.ReadLine()) != null)
                {
                    list.Add(line);
                }
                mapTable = list.ToArray();
            }
        }
        static string Map(string statement)
        {
            for (int i = 0; i < mapTable.Length; i += 2)
            {
                if (statement == mapTable[i])
                {
                    return mapTable[i + 1];
                }
            }

            return "";
        }


        static async Task<object> StreamingMicRecognizeAsync(int seconds)
        {
            if (NAudio.Wave.WaveIn.DeviceCount < 1)
            {
                Console.WriteLine("No microphone!");
                return -1;
            }
            var speech = SpeechClient.Create();
            var streamingCall = speech.StreamingRecognize();
            // Write the initial request with the config.
            await streamingCall.WriteAsync(
                new StreamingRecognizeRequest()
                {
                    StreamingConfig = new StreamingRecognitionConfig()
                    {
                        Config = new RecognitionConfig()
                        {
                            Encoding =
                            RecognitionConfig.Types.AudioEncoding.Linear16,
                            SampleRateHertz = 16000,
                            LanguageCode = "vi",
                        },
                        InterimResults = true,
                    }
                });
            // Print responses as they arrive.
            Task printResponses = Task.Run(async () =>
            {
                while (await streamingCall.ResponseStream.MoveNext(
                    default(CancellationToken)))
                {
                    foreach (var result in streamingCall.ResponseStream
                        .Current.Results)
                    {
                        if (result.IsFinal == true)
                        {
                            s = result.Alternatives[0].Transcript;
                            s = s.ToLower().Trim();
                            //s = Map(s);
                            if (s != "")
                            {
                                Console.WriteLine(s);
                                SendKey(s);
                            }
                            //}

                        }
                    }
                }
            });
            // Read from the microphone and stream to API.
            object writeLock = new object();
            bool writeMore = true;
            var waveIn = new NAudio.Wave.WaveInEvent();
            waveIn.DeviceNumber = 0;
            waveIn.WaveFormat = new NAudio.Wave.WaveFormat(16000, 1);
            waveIn.DataAvailable +=
                (object sender, NAudio.Wave.WaveInEventArgs args) =>
                {
                    lock (writeLock)
                    {
                        if (!writeMore) return;
                        streamingCall.WriteAsync(
                            new StreamingRecognizeRequest()
                            {
                                AudioContent = Google.Protobuf.ByteString
                                    .CopyFrom(args.Buffer, 0, args.BytesRecorded)
                            }).Wait();
                    }
                };
            waveIn.StartRecording();
            Console.WriteLine("Speak now.");
            await Task.Delay(TimeSpan.FromSeconds(seconds));
            // Stop recording and shut down.
            waveIn.StopRecording();
            lock (writeLock) writeMore = false;
            await streamingCall.WriteCompleteAsync();
            await printResponses;
            return 0;
        }

        public static int Main(string[] args)
        {
            if (args.Length != 0)
            {
                program = args[0];
            }
            //SetMapTable();
            Task t = Task.Run(() => StreamingMicRecognizeAsync(10).Result);
            t.Wait();
            return 0;
        }
    }
}
