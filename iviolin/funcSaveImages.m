function funcSaveImages(PathResultImages,fMTF,MTF,fNPS1,NPS1,fNPS2,NPS2)
%04.02.2022
%Save images in specified folder without displaying them (this is required to make this run from console without a GUI)
%1. MTF plot
%2. PS plot
%3. NPS plot
%4. PS/MTF plot

ImgPath = PathResultImages;

%% 1. MTF Plot (f,MTF)
  MTF = MTF/MTF(2); %Normalize so that plot start at y=1
  %Define File Name
  FileName = 'MTF';
  %Generate and adjust plot (not visible)
  h=figure('Position', [100 100 700 700],"visible", "off"); 
  hold on
  grid on
  h1=plot(fMTF(2:end),MTF(2:end));
  xlabel('Spatial frequency [mm^{-1}]'); 
  ylabel('MTF(f) (normalized)');
  xlim([fMTF(2) fMTF(floor(length(fMTF)/2))]);
  ylim([0 1]);
  set(h1, 'LineWidth', 4);
  set(gca, 'Box', 'off' ); 
  set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
  set(gca,'FontSize',12);
  %Save plot as JPG and PNG images 
  print(h,"-S1920,1200",'-djpg',strcat(ImgPath,FileName,'.jpg'));
  print(h,"-S1920,1200",'-dpng',strcat(ImgPath,FileName,'.png'));
  
  
%% 2. PS Plot (fNPS1,NPS1)
  %Define File Name
  FileName = 'PS';
  %Generate and adjust plot (not visible)
  h=figure('Position', [100 100 700 700],"visible", "off"); 
  hold on
  grid on
  h1 = plot(fNPS1,NPS1,'x-');
  xlabel('Spatial frequency [mm^{-1}]'); 
  ylabel('PS [mm^2]');
  xlim([0 fNPS1(end)]);
  ylim([0 max(NPS1)]);
  set(h1, 'LineWidth', 4);
  set(gca, 'Box', 'off' ); 
  set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
  set(gca,'FontSize',12);
  %Save plot as JPG and PNG images 
  print(h,"-S1920,1200",'-djpg',strcat(ImgPath,FileName,'.jpg'));
  print(h,"-S1920,1200",'-dpng',strcat(ImgPath,FileName,'.png'));
  
%% 3. NPS Plot (fNPS2,NPS2)
  %Define File Name
  FileName = 'NPS';
  %Generate and adjust plot (not visible)
  h=figure('Position', [100 100 700 700],"visible", "off"); 
  hold on
  grid on
  h1 = plot(fNPS2,NPS2,'x-');
  xlabel('Spatial frequency [mm^{-1}]'); 
  ylabel('NPS [mm^2]');
  xlim([0 fNPS2(end)]);
  if max(NPS2) == 0 
    ylim([0 1]);
  else
    ylim([0 max(NPS2)]);
  end
  set(h1, 'LineWidth', 4);
  set(gca, 'Box', 'off' ); 
  set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
  set(gca,'FontSize',12);
  %Save plot as JPG and PNG images 
  print(h,"-S1920,1200",'-djpg',strcat(ImgPath,FileName,'.jpg'));
  print(h,"-S1920,1200",'-dpng',strcat(ImgPath,FileName,'.png'));
  
  
%% 4. PS/MTF plot (! PS==NPS1 !)
  %Prepocess data:
  fMTF = fMTF(2:end);
  MTF = MTF(2:end);
  fMin = fMTF(1);
  fMax = fNPS1(end);
  %Interpolate / resample NPS and MTF curves to have the same frequency
  %points and the same number of samples
  f_int = fMin:0.02:fMax;
  MTF_int = interp1(fMTF,MTF,f_int);
  NPS_int = interp1(fNPS1,NPS1,f_int);

  %Define File Name
  FileName = 'PStoMTF';
  %Generate and adjust plot (not visible)
  h=figure('Position', [100 100 700 700],"visible", "off"); 
  hold on
  grid on
  h1=plot(f_int,NPS_int./MTF_int);
  xlabel('Spatial frequency [mm^{-1}]');
  ylabel('Intensity [a.u.]');
  xlim([fMin fMax]);
  set(h1, 'LineWidth', 4);
  set(gca, 'Box', 'off' ); 
  set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
  set(gca,'FontSize',12);
  %Save plot as JPG and PNG images 
  print(h,"-S1920,1200",'-djpg',strcat(ImgPath,FileName,'.jpg'));
  print(h,"-S1920,1200",'-dpng',strcat(ImgPath,FileName,'.png'));
  