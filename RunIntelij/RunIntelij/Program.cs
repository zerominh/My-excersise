using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace RunIntelij
{
    class Program
    {
        [DllImport("User32.dll")]
        static extern int SetForegroundWindow(IntPtr point);
        static void Main(string[] args)
        {
            Process p = Process.Start(@"C:\Program Files\JetBrains\IntelliJ IDEA 2017.3\bin\idea64.exe");
            p.WaitForInputIdle();
        }
    }
}
