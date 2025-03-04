function NPSresult = funcCalcNPScompleteROI(IMG,px,ImShowSizeX, ImShowSizeY, ROIname,PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE,HOST_TYPE)
%% Function summary
% Date: 29.03.2021
% Compute the NPS for the provided ROI without selecting a homogneous area

% Definitions: 
% 

% Function Inputs:
%   ...
%   ImagePathInfo - Contains the complete image name and path info (used later for automatically save the results)
%   ROIname - Used as file name extension (eg '...-ROI1.jpg' or '...-ROI2.jpg'


% Function Returns:
%   

% Updates:

NPSpix = IMG;

%% NPS computation

%Deterend the pixel matrix
NPSpix = NPSpix - mean2(NPSpix);
% NPSpix = detrend(NPSpix,2);

NPS_roi_size = length(NPSpix(:,1));

%Variance of the ROI
var_measured = var(NPSpix(:));
disp(['Pixel variance measured with var(): ' num2str((var_measured))])

%Compute the NPS
[nps_measured, f] = funcCalcDigitalNps(NPSpix, 2, px, 1, 0); % No NPS averaging
var_NPS = trapz(f, trapz(f, nps_measured)); % The variance is the integral of the NPS:
disp(['Pixel variance from measured NPS: ' num2str((var_NPS))])

%Radial averaging of the 2D-NPS, Version 1 (i.e. of the 2D-FFT of the noise image)
nps_measured_radagv = radialAverage(nps_measured, (NPS_roi_size+0)/2, (NPS_roi_size+0)/2, [0:1:NPS_roi_size/2-1]);
f_avg = linspace(0,1/px/2,length(nps_measured_radagv));

% [Zr, R] = radialavg(nps_measured,8);
% if DEBUG_PLOT_LEVEL >= 2
%     figure;plot(R,Zr);
% end

%% Summarize results as a structure
NPSresult.nps_measured = nps_measured;
NPSresult.f = f;
NPSresult.nps_measured_radagv = nps_measured_radagv;
NPSresult.f_avg = f_avg;
NPSresult.var_measured = var_measured;
NPSresult.var_NPS = var_NPS;


%% Save results in a txt-file
% ##if HOST_TYPE == 0
% ##  txtPath = strcat('Results\');%strcat(ImagePathInfo.PathResults);
% ##  txtName = strcat('Pat-',int2str(PatientNumber),'-Results-NPS-CompleteROI'); %strcat(ImagePathInfo.FName,'-MTFarea','-',ROIname);
% ##  txtPathFnameFull = strcat(txtPath,txtName);
% ##
% ##  fileID = fopen(strcat(txtPathFnameFull,'.txt'),'w');
% ##  fprintf(fileID,'-------------------------------------------------------\n');
% ##  if ENV_TYPE == 0 %MATLAB
% ##    fprintf(fileID,'Results timestamp: %s\n',datetime);
% ##  end  
% ##  fprintf(fileID,'Results for Dataset: %s  ## image %s (Patient Number used in Matlab: %d) \n \n',ImagePathInfo.SubFolderName, ImagePathInfo.FName, PatientNumber);
% ##  %fprintf(fileID,'ROI size= %3.0f x %3.0f pixel / ROI center positon (X,Y): (%3.0f , %3.0f) \n',ROI_SIZE, ROI_SIZE, ROI_X, ROI_Y);
% ##  fprintf(fileID,'!ATTENTION: the complete ROi was used to compute the NPS (same ROI as used for MTF)! \n',var_measured);
% ##  fprintf(fileID,'Pixel variance measured with var() = %6.0f \n',var_measured);
% ##  fprintf(fileID,'Pixel variance measured with NPS() = %6.0f \n',var_NPS);
% ##
% ##  fprintf(fileID,'\n \n \n');
% ##  fclose(fileID);
% ##end


%% PLOT - 1D NPS obtained from radial avaerging
if DEBUG_PLOT_LEVEL >= 1
    h=figure('Position', [100 100 700 700]);
    h1 = plot(f_avg,nps_measured_radagv,'x-');
    grid on
    xlabel('Frequency [mm^{-1}]'); 
    ylabel('NPS [mm^2]')
    xlim([0 1/px/2])
    ylim([0 max(nps_measured_radagv(:))])
    set(h1, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS1D-CompleteROI');
    imgPathFnameFull = strcat(imgPath,imgName);
    hgexport(h,strcat(imgPathFnameFull,'.eps'));
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end    
    
    
%% PLOT - 2D NPS 
if DEBUG_PLOT_LEVEL >= 1
    h=figure('Position', [100 100 700 700]);
    imagesc(f,f,abs(nps_measured(:,:,1)))
    xlabel('Frequency [mm^{-1}]'); 
    ylabel('Frequency [mm^{-1}]');
    set(gca,'View',[-0 -90]) %Rotate the figure
    %set(gca,'visible','off') %Hide axes
    xlim([-1/px/2 1/px/2])
    ylim([-1/px/2 1/px/2])
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);
    
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS2D-CompleteROI');
    imgPathFnameFull = strcat(imgPath,imgName);
    hgexport(h,strcat(imgPathFnameFull,'.eps'));
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

return %End of this function