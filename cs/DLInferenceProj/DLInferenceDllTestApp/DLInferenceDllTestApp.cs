using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DLInferenceDllRev;

namespace DLInferenceDllTestApp

{
    //[TestClass]
    public class DLInferenceDllTestApp
    {
       // [TestMethod]

        static void Main(string[] args) {
            DLInferenceRev dl_infer_rev = new DLInferenceRev("D:\\projs\\models\\model_Training-240827-142232_opt.hdl");
            //dl_infer_rev.testc();
            string str_InferredResults;
            dl_infer_rev.ProcessImgWithDLModel("D:\\projs\\data\\A¼¶Ãæ\\28.bmp", out str_InferredResults);
            Console.WriteLine($"28.bmp info:{str_InferredResults}");
            Console.ReadKey();
        }
    }

}

//using  System;
//static void Main(string[] args) {
//    Console.WriteLine("hello TestDllInferenceDll Main()");
//    Console.ReadLine();

//}