function structNpsMtfRatioResults = funcRatioNpsMtf(fNPS,NPS,fMTF,MTF,PatientNumber,imgROI1,DEBUG_PLOT_LEVEL,ENV_TYPE)

fMTF = fMTF(2:end);
MTF = mean(MTF(2:end,:)');

fMin = fMTF(1);
fMax = fNPS(end);

%Interpolate / resample NPS and MTF curves to have the same frequency
%points and the same number of samples
f_int = fMin:0.01:fMax;
MTF_int = interp1(fMTF,MTF,f_int);
NPS_int = interp1(fNPS,NPS,f_int);

NPS_int_scale = NPS_int/NPS_int(1);    %%%%%%%%%%%

%% 15.01.2022: Analyse the NPS/MTF ratio curve to determine the image quality
ratioNpsMtf = NPS_int./MTF_int;

% Analysis 1: Ratio of area under curve between LF/HF component (divide at
% center frequency
fc = round(length(ratioNpsMtf)/2);
areaLf=trapz(ratioNpsMtf(1:fc));
areaHf=trapz(ratioNpsMtf(fc+1:end));
ratioLfHf = areaLf/areaHf;

% Analysis 2: Slope of the NPS/MTF ratio -> average
slopeAvg = mean(diff(ratioNpsMtf));

% Analysis 3: Location of maximum (given in % of the frequency axis)
[val,pos] = max(ratioNpsMtf);
locMax = pos/length(ratioNpsMtf);

% Save all results in a strucutre
structNpsMtfRatioResults.ratioLfHf = ratioLfHf;
structNpsMtfRatioResults.slopeAvg = slopeAvg;
structNpsMtfRatioResults.locMax = locMax;
structNpsMtfRatioResults.ratioNpsMtf = ratioNpsMtf; %Data points
structNpsMtfRatioResults.fInt = f_int; %Frequency values for the "ratioNpsMtf" data points


%% Plot results
if DEBUG_PLOT_LEVEL >= 1 %|| DEBUG_PLOT_LEVEL == -1 
    h=figure('Position', [10 10 610 610]); hold on;grid on
    h1=plot(f_int,MTF_int,'o')
    h2=plot(f_int,NPS_int_scale,'+')
    h3=plot(f_int,NPS_int_scale./MTF_int,'d');
    h4=plot(f_int,sqrt(NPS_int_scale)./MTF_int,'^');
    %imshow(myimg);colormap(gray);truesize([ImShowSizeX ImShowSizeY])
    xlabel('Spatial frequency [1/mm]')
    ylabel('Intensity [a.u.]')
    xlim([fMin fMax])
    set(h1, 'LineWidth', 2, 'color', [0 0 0]);
    set(h2, 'LineWidth', 2, 'color', [0.0 0.0 1]);
    set(h3, 'LineWidth', 2);
    set(h4, 'LineWidth', 2);
    set(gca, 'Box', 'off' );
    set(gca, 'TickDir', 'out','LineWidth',2 ,'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);
    legend('MTF','PS','PS/MTF','sqrt(PS)/MTF','location','northwest')
    
    if DEBUG_PLOT_LEVEL >= 1
        %Export image as EPS and JPG and PNG
        imgPath = strcat('Images\');
        imgName = strcat('Pat-',int2str(PatientNumber),'-Ratio-NPS-MTF');
        imgPathFnameFull = strcat(imgPath,imgName);
        if ENV_TYPE == 0 %MATLAB
            hgexport(h,strcat(imgPathFnameFull,'.eps'));
        end
        saveas(h,strcat(imgPathFnameFull,'.png'));
        print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    end
end


%% 26.04.21: Suggestion CH: Only NPS/MTF ratio plot    
if DEBUG_PLOT_LEVEL >= 1 || DEBUG_PLOT_LEVEL == -1
    
    h=figure('Position', [10 10 610 610]); hold on;grid on
    h1=plot(f_int,NPS_int./MTF_int,'o')
    xlabel('Spatial frequency [1/mm]')
    ylabel('Intensity [a.u.]')
    xlim([fMin fMax])
    set(h1, 'LineWidth', 2, 'color', [0 0 0]);
    set(gca, 'Box', 'off' );
    set(gca, 'TickDir', 'out','LineWidth',2 ,'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);
    legend('PS/MTF','location','northwest')
    
    if DEBUG_PLOT_LEVEL >= 1
        %Export image as EPS and JPG and PNG
        imgPath = strcat('Images\');
        imgName = strcat('Pat-',int2str(PatientNumber),'-Ratio-PS-MTF-NoNorm');
        imgPathFnameFull = strcat(imgPath,imgName);
        if ENV_TYPE == 0 %MATLAB
            hgexport(h,strcat(imgPathFnameFull,'.eps'));
        end
        saveas(h,strcat(imgPathFnameFull,'.png'));
        print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    end
end

if DEBUG_PLOT_LEVEL >= 1 %|| DEBUG_PLOT_LEVEL == -1
    h=figure('Position', [10 10 610 610]); hold on;grid on
    h1=plot(f_int,sqrt(NPS_int)./MTF_int,'o')
    xlabel('Spatial frequency [1/mm]')
    ylabel('Intensity [a.u.]')
    xlim([fMin fMax])
    set(h1, 'LineWidth', 2, 'color', [0 0 0]);
    set(gca, 'Box', 'off' );
    set(gca, 'TickDir', 'out','LineWidth',2 ,'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);
    legend('sqrt(PS)/MTF','location','northwest')
    
    if DEBUG_PLOT_LEVEL >= 1
        %Export image as EPS and JPG and PNG
        imgPath = strcat('Images\');
        imgName = strcat('Pat-',int2str(PatientNumber),'-Ratio-sqrtPS-MTF-NoNorm');
        imgPathFnameFull = strcat(imgPath,imgName);
        if ENV_TYPE == 0 %MATLAB
            hgexport(h,strcat(imgPathFnameFull,'.eps'));
        end
        saveas(h,strcat(imgPathFnameFull,'.png'));
        print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    end
end


end %End of function