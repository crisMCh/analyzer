function funcSaveResultsTxtFile(PathResultFile,info,MTFarea,imgROI1_NPS,imgROI2_NPS,structNpsMtfRatioResults, pred)
%04.02.2022

%PathResultFile ... Path where result should be stored
%info ... DICOM header

%Save all result values in a txt-File

%Concatenate Path and file name
FileName = 'result.txt';
txtPath = strcat(PathResultFile,FileName);

%Parameters
dicomFileName = info.Filename;

%Save date to file 
fileID = fopen(txtPath,'w');
fprintf(fileID,'-------------------------------------------------------\n');
fprintf(fileID,'DICOM File Name: %s \n',dicomFileName);
%MTF
fprintf(fileID,'-------------------------------------------------------\n');
fprintf(' \n')
fprintf(fileID,'\n--------------------MTF--------------------------------\n');
fprintf(fileID,'[MTF] Mean area under MTF curve  = %6.3f \n',mean(MTFarea));

%PS
fprintf(fileID,'\n--------------------PS---------------------------------\n');
fprintf(fileID,'[PS] Pixel variance measured with var() = %6.0f \n',imgROI1_NPS.var_measured);
fprintf(fileID,'[PS] Pixel variance measured with NPS() = %6.0f \n',imgROI1_NPS.var_NPS);

%NPS
fprintf(fileID,'\n--------------------NPS--------------------------------\n');
fprintf(fileID,'[NPS] Pixel variance measured with var() = %6.0f \n',imgROI2_NPS.var_measured);
fprintf(fileID,'[NPS] Pixel variance measured with NPS() = %6.0f \n',imgROI2_NPS.var_NPS);

%PS/MTF ratio
fprintf(fileID,'\n--------------------PS/MTF-----------------------------\n');
fprintf(fileID,'[PS/MTF] LF/HF ratio of area under the curve = %6.2f \n',structNpsMtfRatioResults.ratioLfHf);
fprintf(fileID,'[PS/MTF] Average slope of the curve = %6.1f \n',structNpsMtfRatioResults.slopeAvg);
fprintf(fileID,'[PS/MTF] Location of the maximum (as percentage of the Frequency axis) = %1.2f \n',structNpsMtfRatioResults.locMax); 

%Prediction label and Probability estimate
fprintf(fileID,'\n--------------------Label Prediction-----------------------------\n');
fprintf(fileID,'Prediction label = %d \n', pred.label);
fprintf(fileID,'Decision value/Estimate of probability = %.2f \n', pred.prob);

%Close file
fclose(fileID);  
  
end
