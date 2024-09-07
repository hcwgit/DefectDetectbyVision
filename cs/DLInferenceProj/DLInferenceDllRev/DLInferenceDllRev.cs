using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HalconDotNet;
using System.Web;
using static DLInferenceDllRev.DLInferenceRev;

namespace DLInferenceDllRev
{


    public class DLInferenceRev
    {
        private HTuple hv_ModelID;
        public struct ModelParams {
          public HTuple hv_model_width;
          public HTuple hv_model_height;
          public HTuple hv_model_precision;
          public HTuple hv_nums_channel;
          public HTuple hv_model_type;
        }
        ModelParams hv_ModelParams;

        public  DLInferenceRev(string model_path)
        {
            if (model_path.Length == 0)
            {
                Console.WriteLine("model path is NULL");
                return;
            }

            if (!File.Exists(model_path))
            {
                Console.WriteLine("the path:\n {model_path} is invalid!");
                return;
            }

            if (hv_ModelID == null)
            {
                hv_ModelID = new HTuple();
            }

            try
            {
                HOperatorSet.ReadDlModel(model_path, out hv_ModelID);
            }
            catch (HalconException ex){
                Console.WriteLine(ex.Message);
            }
           
        }

        //for test
        public void testc() {
            Console.WriteLine("DLInferenceDllRev::testc is invoked!");
        }

        //get the params of the model with the file extension ".hdl"
        public void GetModelParams(HTuple hv_model_id, out ModelParams model_params)
        {
            model_params = new ModelParams();
            if (hv_model_id == null) {
                Console.WriteLine("the model id is null!");
                return;
            }

            HOperatorSet.GetDlModelParam(hv_model_id, "image_height", out model_params.hv_model_height);
            HOperatorSet.GetDlModelParam(hv_model_id, "image_width", out model_params.hv_model_width);
            HOperatorSet.GetDlModelParam(hv_model_id, "image_num_channels", out model_params.hv_nums_channel);
            HOperatorSet.GetDlModelParam(hv_model_id, "precision", out model_params.hv_model_precision);
            HOperatorSet.GetDlModelParam(hv_model_id, "type", out model_params.hv_model_type);
            return;
        }
        // Chapter: Deep Learning / Anomaly Detection and Global Context Anomaly Detection
        // Short Description: Apply the given thresholds on anomaly detection and Global Context Anomaly Detection results for image classification and region segmentation. 
        public void threshold_dl_anomaly_results(HTuple hv_AnomalySegmentationThreshold,
            HTuple hv_AnomalyClassificationThreshold, HTuple hv_DLResults)
        {



            // Local iconic variables 

            HObject ho_AnomalyImage = null, ho_AnomalyRegion = null;

            // Local control variables 

            HTuple hv_DLResultIndex = new HTuple(), hv_DLResult = new HTuple();
            HTuple hv_DLResultKeys = new HTuple(), hv_ImageKeys = new HTuple();
            HTuple hv_ScoreKeys = new HTuple(), hv_Index = new HTuple();
            HTuple hv_AnomalyImageRegionType = new HTuple(), hv_AnomalyRegionName = new HTuple();
            HTuple hv_AnomalyScoreType = new HTuple(), hv_AnomalyScoreName = new HTuple();
            HTuple hv_AnomalyScore = new HTuple(), hv_AnomalyClassName = new HTuple();
            HTuple hv_AnomalyClassIDName = new HTuple();
            // Initialize local and output iconic variables 
            HOperatorSet.GenEmptyObj(out ho_AnomalyImage);
            HOperatorSet.GenEmptyObj(out ho_AnomalyRegion);
            try
            {
                //This procedure applies given thresholds on anomaly detection (AD)
                //or Global Context Anomaly Detection (GC-AD) results.
                //The thresholds are used for:
                //
                //1. Region segmentation: AnomalySegmentationThreshold is used as threshold
                //whether a pixel within the anomaly image belongs to a region of an anomaly.
                //The region is returned in DLResult under one of the following keys, depending on
                //the anomaly image key stored in the DLResult:
                //- 'anomaly_region' (AD, GC_AD)
                //- 'anomaly_region_local' (GC-AD)
                //- 'anomaly_region_global' (GC-AD)
                //2. Image classification: AnomalyClassificationThreshold is used as threshold
                //whether the image is classified as containing an anomaly ('nok' / class_id: 1) or not ('ok' / class_id: 0).
                //The class is returned in DLResult under one of the following keys:
                //- 'anomaly_class' (AD, GC_AD): The classification result as a string ('ok' or 'nok').
                //- 'anomaly_class_local' (GC-AD): The classification result as a string ('ok' or 'nok').
                //- 'anomaly_class_global' (GC-AD): The classification result as a string ('ok' or 'nok').
                //- 'anomaly_class_id' (AD, GC_AD): The classification result as an integer (0 or 1).
                //- 'anomaly_class_id_local' (GC-AD): The classification result as an integer (0 or 1).
                //- 'anomaly_class_id_global' (GC-AD): The classification result as an integer (0 or 1).
                //
                //The applied thresholds are also stored in DLResult.
                //
                //Check for invalid AnomalySegmentationThreshold.
                if ((int)(new HTuple((new HTuple(hv_AnomalySegmentationThreshold.TupleLength()
                    )).TupleNotEqual(1))) != 0)
                {
                    throw new HalconException("AnomalySegmentationThreshold must be specified by exactly one value.");
                }
                //
                //Check for invalid AnomalyClassificationThreshold.
                if ((int)(new HTuple((new HTuple(hv_AnomalyClassificationThreshold.TupleLength()
                    )).TupleNotEqual(1))) != 0)
                {
                    throw new HalconException("AnomalyClassificationThreshold must be specified by exactly one value.");
                }
                //
                //Evaluate each DLResult.
                for (hv_DLResultIndex = 0; (int)hv_DLResultIndex <= (int)((new HTuple(hv_DLResults.TupleLength()
                    )) - 1); hv_DLResultIndex = (int)hv_DLResultIndex + 1)
                {
                    //
                    //Read anomaly image and anomaly score from DLResult.
                    hv_DLResult.Dispose();
                    using (HDevDisposeHelper dh = new HDevDisposeHelper())
                    {
                        hv_DLResult = hv_DLResults.TupleSelect(
                            hv_DLResultIndex);
                    }
                    hv_DLResultKeys.Dispose();
                    HOperatorSet.GetDictParam(hv_DLResult, "keys", new HTuple(), out hv_DLResultKeys);
                    hv_ImageKeys.Dispose();
                    HOperatorSet.TupleRegexpSelect(hv_DLResultKeys, ".*_image.*", out hv_ImageKeys);
                    hv_ScoreKeys.Dispose();
                    HOperatorSet.TupleRegexpSelect(hv_DLResultKeys, ".*_score.*", out hv_ScoreKeys);
                    {
                        HTuple ExpTmpOutVar_0;
                        HOperatorSet.TupleSort(hv_ImageKeys, out ExpTmpOutVar_0);
                        hv_ImageKeys.Dispose();
                        hv_ImageKeys = ExpTmpOutVar_0;
                    }
                    {
                        HTuple ExpTmpOutVar_0;
                        HOperatorSet.TupleSort(hv_ScoreKeys, out ExpTmpOutVar_0);
                        hv_ScoreKeys.Dispose();
                        hv_ScoreKeys = ExpTmpOutVar_0;
                    }
                    if ((int)((new HTuple(hv_ImageKeys.TupleEqual(new HTuple()))).TupleOr(new HTuple(hv_ScoreKeys.TupleEqual(
                        new HTuple())))) != 0)
                    {
                        throw new HalconException(new HTuple("DLResult must contain keys 'anomaly_image' (local, global) and 'anomaly_score' (local, global)."));
                    }
                    for (hv_Index = 0; (int)hv_Index <= (int)((new HTuple(hv_ImageKeys.TupleLength()
                        )) - 1); hv_Index = (int)hv_Index + 1)
                    {
                        //Apply AnomalyThreshold to the anomaly image.
                        using (HDevDisposeHelper dh = new HDevDisposeHelper())
                        {
                            ho_AnomalyImage.Dispose();
                            HOperatorSet.GetDictObject(out ho_AnomalyImage, hv_DLResult, hv_ImageKeys.TupleSelect(
                                hv_Index));
                        }
                        ho_AnomalyRegion.Dispose();
                        HOperatorSet.Threshold(ho_AnomalyImage, out ho_AnomalyRegion, hv_AnomalySegmentationThreshold,
                            "max");
                        //
                        //Write AnomalyRegion to DLResult.
                        using (HDevDisposeHelper dh = new HDevDisposeHelper())
                        {
                            hv_AnomalyImageRegionType.Dispose();
                            HOperatorSet.TupleRegexpMatch(hv_ImageKeys.TupleSelect(hv_Index), "anomaly_image(.*)",
                                out hv_AnomalyImageRegionType);
                        }
                        hv_AnomalyRegionName.Dispose();
                        using (HDevDisposeHelper dh = new HDevDisposeHelper())
                        {
                            hv_AnomalyRegionName = "anomaly_region" + hv_AnomalyImageRegionType;
                        }
                        HOperatorSet.SetDictObject(ho_AnomalyRegion, hv_DLResult, hv_AnomalyRegionName);
                        //
                        //Classify sample as 'ok' or 'nok'.
                        using (HDevDisposeHelper dh = new HDevDisposeHelper())
                        {
                            hv_AnomalyScoreType.Dispose();
                            HOperatorSet.TupleRegexpMatch(hv_ScoreKeys.TupleSelect(hv_Index), "anomaly_score(.*)",
                                out hv_AnomalyScoreType);
                        }
                        hv_AnomalyScoreName.Dispose();
                        using (HDevDisposeHelper dh = new HDevDisposeHelper())
                        {
                            hv_AnomalyScoreName = hv_ScoreKeys.TupleSelect(
                                hv_Index);
                        }
                        hv_AnomalyScore.Dispose();
                        HOperatorSet.GetDictTuple(hv_DLResult, hv_AnomalyScoreName, out hv_AnomalyScore);
                        hv_AnomalyClassName.Dispose();
                        using (HDevDisposeHelper dh = new HDevDisposeHelper())
                        {
                            hv_AnomalyClassName = "anomaly_class" + hv_AnomalyScoreType;
                        }
                        hv_AnomalyClassIDName.Dispose();
                        using (HDevDisposeHelper dh = new HDevDisposeHelper())
                        {
                            hv_AnomalyClassIDName = "anomaly_class_id" + hv_AnomalyScoreType;
                        }
                        if ((int)(new HTuple(hv_AnomalyScore.TupleLess(hv_AnomalyClassificationThreshold))) != 0)
                        {
                            HOperatorSet.SetDictTuple(hv_DLResult, hv_AnomalyClassName, "ok");
                            HOperatorSet.SetDictTuple(hv_DLResult, hv_AnomalyClassIDName, 0);
                        }
                        else
                        {
                            HOperatorSet.SetDictTuple(hv_DLResult, hv_AnomalyClassName, "nok");
                            HOperatorSet.SetDictTuple(hv_DLResult, hv_AnomalyClassIDName, 1);
                        }
                    }
                    //
                    //Write anomaly thresholds to DLResult.
                    HOperatorSet.SetDictTuple(hv_DLResult, "anomaly_classification_threshold",
                        hv_AnomalyClassificationThreshold);
                    HOperatorSet.SetDictTuple(hv_DLResult, "anomaly_segmentation_threshold",
                        hv_AnomalySegmentationThreshold);
                }
                //
                ho_AnomalyImage.Dispose();
                ho_AnomalyRegion.Dispose();

                hv_DLResultIndex.Dispose();
                hv_DLResult.Dispose();
                hv_DLResultKeys.Dispose();
                hv_ImageKeys.Dispose();
                hv_ScoreKeys.Dispose();
                hv_Index.Dispose();
                hv_AnomalyImageRegionType.Dispose();
                hv_AnomalyRegionName.Dispose();
                hv_AnomalyScoreType.Dispose();
                hv_AnomalyScoreName.Dispose();
                hv_AnomalyScore.Dispose();
                hv_AnomalyClassName.Dispose();
                hv_AnomalyClassIDName.Dispose();

                return;
            }
            catch (HalconException HDevExpDefaultException)
            {
                ho_AnomalyImage.Dispose();
                ho_AnomalyRegion.Dispose();

                hv_DLResultIndex.Dispose();
                hv_DLResult.Dispose();
                hv_DLResultKeys.Dispose();
                hv_ImageKeys.Dispose();
                hv_ScoreKeys.Dispose();
                hv_Index.Dispose();
                hv_AnomalyImageRegionType.Dispose();
                hv_AnomalyRegionName.Dispose();
                hv_AnomalyScoreType.Dispose();
                hv_AnomalyScoreName.Dispose();
                hv_AnomalyScore.Dispose();
                hv_AnomalyClassName.Dispose();
                hv_AnomalyClassIDName.Dispose();

                throw HDevExpDefaultException;
            }
        }

        //convert the type of input image to the precision of model for inferring
        public void ConvertImageType(HObject input_image, out HObject out_image, HTuple precision)
        {
            out_image = new HObject();
            //switch (precision.ToString().Equals())
            //{
            //    case "float32":
            //        HOperatorSet.ConvertImageType(input_image, out out_image, "real");
            //        break;
            //    case "byte":
            //        HOperatorSet.ConvertImageType(input_image, out out_image, "byte");
            //        break;
            //    default:
            //        out_image = input_image.Clone();
            //        break;
            //}
            if (!precision.ToString().Equals("float32"))
            {
                HOperatorSet.ConvertImageType(input_image, out out_image, "real");
                //HOperatorSet.ConvertImageType(input_image, out out_image, "float");
                //HOperatorSet.ScaleImage(input_image, out out_image, 1.0, 0.0);
            }
            else if (!precision.ToString().Equals("byte"))
            {
                HOperatorSet.ConvertImageType(input_image, out out_image, "byte");
            }else
            {
                out_image = input_image.Clone();
            }
            return;
        }

        //resize the size of input image to specific size.
        void resizeImage(HObject ho_InputImage, out HObject ho_ImageScaled, HTuple ho_Width, HTuple ho_Height, bool b_ConstantAspectRatio=true, string str_Interpolation = "constant")
        {
            ho_ImageScaled = new HObject();
            if (ho_InputImage == null) {
                Console.WriteLine("reizeImage error: the path of input image is invalid");
                return; 
            }
            HTuple hv_Interpolation = new HTuple(str_Interpolation);

            //get the width and height of input image
            HTuple hv_InputWidth, hv_InputHeight;
            HOperatorSet.GetImageSize(ho_InputImage, out hv_InputWidth, out hv_InputHeight);

            //caculate the aspectRatio of input image and target image
            HTuple hv_AspectRatio = hv_InputWidth / hv_InputHeight;
            HTuple hv_targetAspectRatio = ho_Width / ho_Height;
            if (b_ConstantAspectRatio)
            {
                //保持原图的宽高比
                //float f_aspectRation =  (float)ho_Width/ (float)ho_Height;

                //calculate the scale factor
                // float f_targetWidth = (float)ho_Width;
                // float f_targetHeight = (float)ho_Height;            
                //float targetAspectRatio = f_targetWidth/ f_targetHeight;
                // float newWidth = 0.0f;
                // float newHeight = 0.0f;

                HTuple hv_NewWidth, hv_NewHeight;
                if (hv_targetAspectRatio > hv_AspectRatio)
                {
                    //adjust according to the target height
                    HOperatorSet.TupleRound(ho_Height*hv_AspectRatio, out hv_NewWidth);
                    hv_NewHeight = ho_Height;
                }
               else
                {
                    //adjust according to the target width
                    hv_NewWidth = ho_Width;
                    HOperatorSet.TupleRound(ho_Width, out hv_NewHeight);
                }

                //根据计算得出的宽和高拉伸
                HOperatorSet.ZoomImageSize(ho_InputImage, out ho_ImageScaled, hv_NewWidth, hv_NewHeight, hv_Interpolation);
                
            }
            else
            {
                //根据目标宽高比拉伸
                HOperatorSet.ZoomImageSize(ho_InputImage, out ho_ImageScaled, ho_Width, ho_Height, hv_Interpolation);
            }
            
            hv_Interpolation.Dispose();
            return;
        }

        public string ProcessImgWithDLModel(string img_path, out string str_Results)
        {
            str_Results = null;
            if (hv_ModelID.TupleLength() == 0)
                return string.Format("the model of deep learning is not loaded!");

            if (img_path == null)
                return string.Format("the path of image is non_exist");


            //get the model params;
            ModelParams modelParams = new ModelParams();
            GetModelParams(hv_ModelID, out modelParams);
            Console.WriteLine($"modelParams.hv_model_type: {modelParams.hv_model_precision}\n" +                         
                              $"modelParams.hv_nums_channel: {modelParams.hv_nums_channel}\n"+
                               $"modelParams.hv_model_type: {modelParams.hv_model_type}\n" +
                              $"modelParams.hv_model_width: {modelParams.hv_model_width}\n" +
                              $"modelParams.hv_model_height: {modelParams.hv_model_height}\n");

            HObject ho_Image = new HObject();
           
            //存放预测结果
            HTuple hv_Results = new HTuple();
            //read deep learning model (eg:TensorFlow List)
            HTuple hv_ChannelCount;
            HTuple hv_MetaData = new HTuple();
            HTuple hv_InferenceClassificationThreshold = new HTuple();
            HTuple hv_InferenceSegmentationThreshold = new HTuple();
            HTuple hv_WindowDict = new HTuple();
            HTuple hv_DLDatasetInfo = new HTuple();
            try
            {
                //Create dictionary with dataset parameters used for display.
                hv_DLDatasetInfo.Dispose();
                HOperatorSet.CreateDict(out hv_DLDatasetInfo);
                HOperatorSet.SetDictTuple(hv_DLDatasetInfo, "class_names", (new HTuple("ok")).TupleConcat(
                    "nok"));
                HOperatorSet.SetDictTuple(hv_DLDatasetInfo, "class_ids", (new HTuple(0)).TupleConcat(
                    1));
                //
                hv_WindowDict.Dispose();
                HOperatorSet.CreateDict(out hv_WindowDict);

                //read the image
                HOperatorSet.ReadImage(out ho_Image, img_path);

                //check the num of channels
                HOperatorSet.CountChannels(ho_Image, out hv_ChannelCount);
                
                //check the channel num between model and input image for inferring
                if (hv_ChannelCount != modelParams.hv_nums_channel)
                {
                    //convert the channel of image between the model and input image.
                    if (modelParams.hv_nums_channel == 3)
                    {
                        HObject ho_RGBImage = new HObject();
                        HOperatorSet.Compose3(ho_Image, ho_Image, ho_Image, out ho_RGBImage);
                        ho_Image = ho_RGBImage.Clone();
                        ho_RGBImage.Dispose();
                    }
                    else if (modelParams.hv_nums_channel == 1)
                    {
                        HObject ho_GrayImage = new HObject();
                        HObject imageR, imageG, imageB;
                        HOperatorSet.Decompose3(ho_Image, out imageR, out imageG, out imageB);
                        HOperatorSet.Rgb3ToGray(imageR, imageG, imageB, out ho_GrayImage);
                        ho_Image = ho_GrayImage.Clone();
                        ho_GrayImage.Dispose();
                        imageR.Dispose();
                        imageG.Dispose();
                        imageB.Dispose();
                    }
                    
                }

                //for test
                HOperatorSet.CountChannels(ho_Image, out hv_ChannelCount);
                Console.WriteLine($"after converting the num of image is {hv_ChannelCount}");

                //convert the gray value type of input image to the one of model
                HObject ho_Image_for_model = new HObject();
                ConvertImageType(ho_Image, out ho_Image_for_model, modelParams.hv_model_precision);
                ho_Image = ho_Image_for_model.Clone();
                ho_Image_for_model.Dispose();

                //for check the image type
                HTuple image_type;
                HOperatorSet.GetImageType(ho_Image, out image_type);
                Console.WriteLine($"after converting the type of image is {image_type}");
                

                //get width and height of image
                HTuple hv_width;
                HTuple hv_height;
                HOperatorSet.GetImageSize(ho_Image, out hv_width, out hv_height);
                Console.WriteLine($"before resize ho_Image width:{hv_width.ToString()} height:{hv_height.ToString()}");
                //covert type of gray value
                //HObject ho_Image_word = new HObject();
                //HOperatorSet.ConvertImageType(ho_Image, out ho_Image_word, "word");
                //ho_Image = ho_Image_word;

                //resize the width and height of image,keep the ratio between width and height
               HObject ho_ImageScaled = new HObject();
               resizeImage(ho_Image, out ho_ImageScaled, modelParams.hv_model_width, modelParams.hv_model_height);
               ho_Image = ho_ImageScaled.Clone();
               ho_ImageScaled.Dispose();
               
                

                // 将图像添加到样本批次中
                HTuple hv_DLSampleBatch = new HTuple();

                gen_dl_samples_from_images(ho_Image, out hv_DLSampleBatch);

                //Get thresholds for inference. These have been stored along with
                //the model in the meta data above.
                hv_MetaData.Dispose();
                HOperatorSet.GetDlModelParam(hv_ModelID, "meta_data", out hv_MetaData);
                hv_InferenceClassificationThreshold.Dispose();
                //using (HDevDisposeHelper dh = new HDevDisposeHelper())
                //{
                //hv_InferenceClassificationThreshold = ((hv_MetaData.TupleGetDictTuple(
                //    "anomaly_classification_threshold"))).TupleNumber();
                ////}
                hv_InferenceClassificationThreshold = 0.343;


                hv_InferenceSegmentationThreshold.Dispose();
                hv_InferenceSegmentationThreshold = 0.058;
                //using (HDevDisposeHelper dh = new HDevDisposeHelper())
                //{
                //hv_InferenceSegmentationThreshold = ((hv_MetaData.TupleGetDictTuple(
                //    "anomaly_segmentation_threshold"))).TupleNumber();
                //}

               // hv_Results.Dispose();
                //inference the sample.
                HOperatorSet.ApplyDlModel(hv_ModelID, hv_DLSampleBatch, new HTuple(), out hv_Results);

                //Apply thresholds to classify regions and the entire image.
                threshold_dl_anomaly_results(hv_InferenceSegmentationThreshold, hv_InferenceClassificationThreshold,
                    hv_Results);

                //Display the inference result.
                //dev_display_dl_data(hv_DLSampleBatch, hv_Results, hv_DLDatasetInfo, (new HTuple("anomaly_result")).TupleConcat(
                //    "anomaly_image"), new HTuple(), hv_WindowDict);
                str_Results = hv_Results.ToString();
                //HOperatorSet.ApplyDlModel()
            }
            catch (HalconException ex)
            {
                Console.WriteLine($"halcon error: {ex.Message}");
            }
            finally
            {
                ho_Image.Dispose();
                //hv_DLSampleBatch.Dispose();
            }
            return hv_Results.ToString();
        }

        //Chapter: Deep Learning / Model
        //Short Description: Store the given images in a tuple of dictionaries DLSamples.
        public void gen_dl_samples_from_images(HObject ho_Images, out HTuple hv_DLSampleBatch)
        {

            // Local iconic variables 

            HObject ho_Image = null;

            // Local control variables 

            HTuple hv_NumImages = new HTuple(), hv_ImageIndex = new HTuple();
            HTuple hv_DLSample = new HTuple();
            // Initialize local and output iconic variables 
            HOperatorSet.GenEmptyObj(out ho_Image);
            hv_DLSampleBatch = new HTuple();
            try
            {
                //
                //This procedure creates DLSampleBatch, a tuple
                //containing a dictionary DLSample
                //for every image given in Images.
                //
                //Initialize output tuple.
                hv_NumImages.Dispose();
                HOperatorSet.CountObj(ho_Images, out hv_NumImages);
                hv_DLSampleBatch.Dispose();
                using (HDevDisposeHelper dh = new HDevDisposeHelper())
                {
                    hv_DLSampleBatch = HTuple.TupleGenConst(
                        hv_NumImages, -1);
                }
                //
                //Loop through all given images.
                HTuple end_val10 = hv_NumImages - 1;
                HTuple step_val10 = 1;
                for (hv_ImageIndex = 0; hv_ImageIndex.Continue(end_val10, step_val10); hv_ImageIndex = hv_ImageIndex.TupleAdd(step_val10))
                {
                    using (HDevDisposeHelper dh = new HDevDisposeHelper())
                    {
                        ho_Image.Dispose();
                        HOperatorSet.SelectObj(ho_Images, out ho_Image, hv_ImageIndex + 1);
                    }
                    //Create DLSample from image.
                    hv_DLSample.Dispose();
                    HOperatorSet.CreateDict(out hv_DLSample);
                    HOperatorSet.SetDictObject(ho_Image, hv_DLSample, "image");
                    //
                    //Collect the DLSamples.
                    if (hv_DLSampleBatch == null)
                        hv_DLSampleBatch = new HTuple();
                    hv_DLSampleBatch[hv_ImageIndex] = hv_DLSample;
                }
                ho_Image.Dispose();

                hv_NumImages.Dispose();
                hv_ImageIndex.Dispose();
                hv_DLSample.Dispose();

                return;
            }
            catch (HalconException HDevExpDefaultException)
            {
                ho_Image.Dispose();

                hv_NumImages.Dispose();
                hv_ImageIndex.Dispose();
                hv_DLSample.Dispose();

                throw HDevExpDefaultException;
            }
        }

    }
}
