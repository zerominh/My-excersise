using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;
namespace SendKey
{
    class Program
    {
        [DllImport("User32.dll")]
        static extern int SetForegroundWindow(IntPtr point);
        static void SendKey()
        {
            Process p = Process.GetProcessesByName("idea64").FirstOrDefault();
            if (p != null)
            {
                    
                    IntPtr h = p.MainWindowHandle;
                    SetForegroundWindow(h);
                    SendKeys.SendWait("k");
                    Console.ReadLine();
            }
        }
        static void Main(string[] args)
        {
            SendKey();
        }
    }
}
