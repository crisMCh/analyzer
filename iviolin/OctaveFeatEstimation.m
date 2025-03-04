%% Image Quality analysis in Octave
%Example call for DICOM image in folder:
%[f_int,MTF_int,NPS1_int,fNPS2,NPS2,ContrastROI1,ContrastROI2]=funcImageQualityOctave('OVGU29NO-I0000148',18,[319,296],[389,286])
% [f_int,MTF_int, NPS1_int,fNPS2,NPS2,ContrastROI1, ContrastROI2, MTFarea, NPS1area, NPS2area, IMGreturn, dicomManufacturer, dicomExposureTime, dicomTubeCurrent, dicomWindowWidth, dicomWindowCenter] = funcImageQualityOctave('OVGU29NO-I0000148',18,[319,296],[389,286])

% Update 04.02.2022:
% - Include new parameters PS/MTF
% - Save plots as files without displaying them

clc, close all
  
%Disable all warnings in console
warning('off','all');

%% Configuration
% 02.05.21 - Set amount of plots
% 0 == No plots, 1 == Important plots (e.g. final results), 2 == All plots
% -1 == Plots to show in the web application (09.10.2021) --> not working and not used
% -2 == Plots to save without displaying them when running on server (04.02.2022) --> not working and not used
DEBUG_PLOT_LEVEL = -2;
ENV_TYPE = 1; %0==MATLAB 1==OCTAVE (Make sure this is set to "1" when running on the server)
HOST_TYPE = 0; %0==Local system, 1==OVGU Server (Make sure this is set to "1" when running on the server)

ROI_SIZE = 18;

% Load Octave packages if Octave Environment is enabled
if ENV_TYPE == 1
  pkg load dicom
  pkg load image
  pkg load signal
  pkg load statistics %kmeans
  pkg load tablicious
end

featSet = [];
PatientNumber = 0;
PathImages = '.\'; %Main path where DICOM images are stored
SubFolderName='';
SubPathResults = 'Results';

% PATH Defintion: Folder for result image plots (MTF, PS; NPS, PS/MTF)
if HOST_TYPE == 0 %Local PC
  PathResultImages = './';
elseif HOST_TYPE == 1 % OVGU Server
end  PathResultImages ='G:/Cristina/Thesis/analyzer/figs/iviolin/octave/'; %Path to store the results images/plots on the OVGU server


% PATH Defintion: Folder for TXT-File contining result values
if HOST_TYPE == 0 %Local PC
  PathResultFile = './';
elseif HOST_TYPE == 1 % OVGU Server
  PathResultFile ='G:/Cristina/Thesis/analyzer/figs/iviolin/octave/'; 
end


goodCases = [10, 11, 12, 19, 20, 21, 60, 61, 62, 63, 64, 65, 66, 67, 68];
badCases  = [13, 16, 18, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59];
availablePatNums = [goodCases,badCases];
% Load DICOM File
for k=1:length(availablePatNums)
  PatientNumber = availablePatNums(k);
  [FName, Label, ROI1t, ROI2t] = funcLoadDicomFile(PatientNumber);

  %% READ DICOM FILE
  %[IMG] = dicomread(strcat(PathImages,'\',SubFolderName,'\',FName));
  [IMG] = dicomread(FName);
  IMG=double(IMG);

  %info = dicominfo(strcat(PathImages,'\',SubFolderName,'\',FName));
  info = dicominfo(FName);
  px = info.PixelSpacing; %Pixel spacing in mm
  px = px(1);
  WW = info.WindowWidth;
  WL = info.WindowCenter;
  infoDICOM = [px,WW,WL]; %vector containg all the required DICOM information

  %% 21.09.2021: DICOM Infos as return values
  dicomManufacturer = info.Manufacturer;
  dicomModel = info.ManufacturerModelName;
  dicomExposureTime = info.ExposureTime;
  dicomTubeCurrent = []; %info.XRayTubeCurrent;
  dicomWindowWidth = info.WindowWidth;
  dicomWindowCenter = info.WindowCenter;

  %% 16.04.2021: Correct the HU values in the image
  IMG = IMG + info.RescaleIntercept;

  % Image to be returned by this function:
  IMGreturn = IMG - min(IMG(:));
  IMGreturn = IMGreturn/max(IMGreturn(:));

  %Save orignal DICOM image as PNG file (04.02.2022)
  if HOST_TYPE == 0 %Local PC
    imwrite(uint8(IMGreturn*255),"Original.png");
  elseif HOST_TYPE == 1 % Server
    %imwrite(IMG, "/home/amedirad/wed/media/Original.png");
    imwrite(uint8(IMGreturn*255), strcat(PathResultImages,"Original.png"));
  end


  %% ROI and active contour mask definition 
  ROI1 = [ROI1t(1),ROI1t(2),ROI_SIZE]; %319,296
  ROI2 = [ROI2t(1),ROI2t(2),ROI_SIZE]; %389,286
  clear ROI1t ROI2t
  ActContMask.x = 10; 
  ActContMask.y = 10;

  %% Crop the ROI selected above from the chosen image
  %ROI 1
  ROI = ROI1;
  [ROI_X,ROI_Y,ROI_SIZE] = deal( ROI(1), ROI(2), ROI(3) );
  imgROI1 = IMG(ROI_Y-ROI_SIZE/2:ROI_Y+ROI_SIZE/2,ROI_X-ROI_SIZE/2:ROI_X+ROI_SIZE/2); %ROI 1 image

  %ROI 2
  ROI = ROI2;
  [ROI_X,ROI_Y,ROI_SIZE] = deal( ROI(1), ROI(2), ROI(3) );
  imgROI2 = IMG(ROI_Y-ROI_SIZE/2:ROI_Y+ROI_SIZE/2,ROI_X-ROI_SIZE/2:ROI_X+ROI_SIZE/2); %ROI 2 image


  [ImShowSizeX, ImShowSizeY] = deal(400,400);

  %% MTF 
  [fMTF,MTF,MTFarea] = funcCalcMTFNew(imgROI1,ActContMask,infoDICOM(1),'ROI1',PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE,HOST_TYPE); %Only for ROI1 (major fissure)
  MTFareaBak = MTFarea;
  MTFarea = mean(MTFarea);
  %% Perform the NPS analysis

  % Parameter defintion for patch analysis (finding homogneous region):
  % Size of the patches used for differentiating between homogenous and structured areas
  PatchSizeX = 3;
  PatchSizeY = 3;

  %   Size of the ROI required for NPS analysis
  NPS_roi_size = 14;  
  NPS_pix_total = NPS_roi_size*NPS_roi_size;

  % NPS fpr homogenous region
  %imgROI1_NPS = funcCalcNPS(imgROI1,PatchSizeX,PatchSizeY,NPS_roi_size,NPS_pix_total,infoDICOM(1),ImShowSizeX, ImShowSizeY,ImagePathInfo,'ROI1');
  imgROI2_NPS = funcCalcNPS(imgROI2,PatchSizeX,PatchSizeY,NPS_roi_size,NPS_pix_total,infoDICOM(1),ImShowSizeX, ImShowSizeY,'ROI2',PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE,HOST_TYPE);

  % 29.03.2021 - call this function to compute the NPS for the complete ROI
  % without any preprocessing
  % NPS for ROI containing the structure (ROI1)
  imgROI1_NPS = funcCalcNPScompleteROI(imgROI1,infoDICOM(1),ImShowSizeX, ImShowSizeY, 'ROI1',PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE,HOST_TYPE);

  fNPS1 = imgROI1_NPS.f_avg;
  NPS1  = imgROI1_NPS.nps_measured_radagv;

  % Compute and analyse PS/MTF ratio in the ROI containing the structure (ROI1)
  %funcRatioNpsMtf(fNPS1,NPS1,fMTF,MTF,PatientNumber,imgROI1,ENV_TYPE);
  structNpsMtfRatioResults = funcRatioNpsMtf(fNPS1,NPS1,fMTF,MTF,PatientNumber,imgROI1,DEBUG_PLOT_LEVEL,ENV_TYPE);

  %ROI mean value
  ROI1_mean = mean2(imgROI1);

  %% Contrast estimation for ROI1 and ROI2
  %RMS-based contrast estimation
  [ContrastROI1] = funcCalcContrast(imgROI1);
  [ContrastROI2] = funcCalcContrast(imgROI2);

  %Prepare results to return
  fNPS1 = imgROI1_NPS.f_avg;
  NPS1  = imgROI1_NPS.nps_measured_radagv;
  NPS1area = imgROI1_NPS.var_NPS;
  fNPS2 = imgROI2_NPS.f_avg;
  NPS2  = imgROI2_NPS.nps_measured_radagv;
  NPS2area = imgROI2_NPS.var_NPS;

  %Round all values for display on website
  ContrastROI1=round(ContrastROI1*1000)/1000;
  ContrastROI2=round(ContrastROI2*1000)/1000;
  MTFarea=round(MTFarea*1000)/1000;
  NPS1area=round(NPS1area);
  NPS2area=round(NPS2area);

  fMTF = fMTF(2:end); %Dont use the DC component in MTF
  MTF = mean(MTF(2:end,:)');
  fMin = fMTF(1); %min value for the freq axis
  if fNPS1(end) < fMTF(end)
    fMax = fNPS1(end); %max value for the freq axis
  else
    fMax = fMTF(end); %max value for the freq axis
  end
  %Interpolate / resample PS (<==NPS1) and MTF curves to have the same frequency
  %points and the same number of samples
  f_int = fMin:0.01:fMax;
  MTF_int  = interp1(fMTF,MTF,f_int);
  NPS1_int = interp1(fNPS1,NPS1,f_int);


  %Save plots (MTF, PS, NPS) as files (PNG and JPG) (04.02.2022)
  funcSaveImages(PathResultImages,fMTF,MTF,fNPS1,NPS1,fNPS2,NPS2,structNpsMtfRatioResults);

  %Save result values in results.txt (04.02.2022)
  funcSaveResultsTxtFile(PathResultFile,info,MTFarea,imgROI1_NPS,imgROI2_NPS,structNpsMtfRatioResults)
    
  % Integrate classification (02.03.2023)
  mtfArea = MTFarea;
  npsHomo = NPS2area;
  npsStruct = NPS1area;
  varStruct = imgROI1_NPS.var_measured;
  ratioLfHf = structNpsMtfRatioResults.ratioLfHf;
  slopeAvg = structNpsMtfRatioResults.slopeAvg;
  locMax = structNpsMtfRatioResults.locMax
  %qualyFeatTable = table(mtfArea,npsHomo,npsStruct,varStruct,ratioLfHf,slopeAvg,locMax);
  %qualyFeatTable
  featSet(k,:) = [mtfArea,npsHomo,npsStruct,varStruct,ratioLfHf,slopeAvg,locMax];

end

save('OctaveFeatSet.mat', 'featSet');

% Perform classification
% disp ("Using SVM prediction now!")
load('svmModelOctave.mat'); % Loads the model
[predicted_label, accuracy, decision_valuesprob_estimates] = ...
svmpredict(1, ...
    featSet, ...
    model);

return





