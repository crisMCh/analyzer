function [EdgeLineProfile,edgePixels] = funcLineProfileNew(myimg,x,y,Gdir,vecLen,LPpoints,imgxsize,imgysize,PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE)

EdgeLineProfile = [];
edgePixels = []; %02.03.2021: Edge pixels which are considered
m=1; %Loop counter variable
for k=1:1:length(x)
    
    if x(k) > vecLen && y(k) > vecLen && ...
            x(k) + vecLen < imgxsize && y(k) + vecLen < imgysize
        
        %Define vector directions according to angles
        gradvecx = cosd(Gdir(y(k),x(k)));
        gradvecy = sind(Gdir(y(k),x(k)));
        %Normalize this vector (actually, with the previous values its already normalized)
        gradvecnorm = [gradvecx,gradvecy]/norm([gradvecx,gradvecy]);
        
        %Define start and end point for the line profiles to be estimated
        vecStart(1,1) = round(x(k)-vecLen*gradvecnorm(1));
        vecStart(1,2) = round(y(k)-(-1*vecLen*gradvecnorm(2))); %Invert direction (-1*) due to rows start counting from top, not from bottom
        vecEnd(1,1)   = round(x(k)+vecLen*gradvecnorm(1));
        vecEnd(1,2)   = round(y(k)+(-1*vecLen*gradvecnorm(2))); %Invert direction (-1*) due to rows start counting from top, not from bottom
        
        if DEBUG_PLOT_LEVEL >= 2
            %Plot lines in previous X-ray image figure:
            h1=line([vecStart(1,1) vecEnd(1,1)], [vecStart(1,2) vecEnd(1,2)],...
                'LineWidth',2,'Color',[1 0 0]);
            %h1.Color=[1 0 0];
        end
        %Extract line profile (Edge Scan Function - ESF)
        if ENV_TYPE == 0 %MATLAB
          EdgeLineProfile(:,m) = improfile(myimg,[vecStart(1,1) vecEnd(1,1)],[vecStart(1,2) vecEnd(1,2)],LPpoints);
        elseif ENV_TYPE == 1 %OCTAVE_EXEC_HOME
          lpTmp = [];
          lpTmp = improfileoct(myimg,[vecStart(1,1) vecEnd(1,1)],[vecStart(1,2) vecEnd(1,2)],LPpoints);
          lpTmpLen = length(lpTmp);
          if mean(lpTmp(1:round(0.25*lpTmpLen))) > mean(lpTmp(round(0.75*lpTmpLen):end))
              lpTmp = flipud(lpTmp);
          end
          EdgeLineProfile(:,m) = lpTmp;
          
        end
        edgePixels(m,1:4) = [x(k),y(k),m,k]; %02.03.2021
        
        m=m+1;
    end %if
end %for

%% Plot 
if DEBUG_PLOT_LEVEL >= 1
    h2=figure;
    imagesc(myimg);colormap(gray);
    if ENV_TYPE == 0, truesize([600 600]), end
    m=1; %Loop counter variable
    vecLen = 5;
    for k=1:4:length(x)

        if x(k) > vecLen & y(k) > vecLen & ...
                x(k) + vecLen < imgxsize & y(k) + vecLen < imgysize

            %Define vector directions according to angles
            gradvecx = cosd(Gdir(y(k),x(k)));
            gradvecy = sind(Gdir(y(k),x(k)));
            %Normalize this vector (actually, with the previous values its already normalized)
            gradvecnorm = [gradvecx,gradvecy]/norm([gradvecx,gradvecy]);

            %Define start and end point for the line profiles to be estimated
            vecStart(1,1) = round(x(k)-vecLen*gradvecnorm(1));
            vecStart(1,2) = round(y(k)-(-1*vecLen*gradvecnorm(2))); %Invert direction (-1*) due to rows start counting from top, not from bottom
            vecEnd(1,1)   = round(x(k)+vecLen*gradvecnorm(1));
            vecEnd(1,2)   = round(y(k)+(-1*vecLen*gradvecnorm(2))); %Invert direction (-1*) due to rows start counting from top, not from bottom

            %Plot lines in previous X-ray image figure:
            h1=line([vecStart(1,1) vecEnd(1,1)], [vecStart(1,2) vecEnd(1,2)],...
                'LineWidth',4);
            if ENV_TYPE == 0, h1.Color=[1 1 0], end

            %Plot rectangles for the relevant edge pixels as well
            rectangle('Position',[x(k)-0.5,y(k)-0.5,1,1],'LineWidth',3,'EdgeColor','g')

            m = m + 1;
            if m>16
                return
            end
        end %if
    end %for

    %Export image as EPS and JPG and PNG
        imgPath = strcat('Images\');
        imgName = strcat('Pat-',int2str(PatientNumber),'-MTF', '-ROI-LineProfile-Overlay');
        imgPathFnameFull = strcat(imgPath,imgName);
        if ENV_TYPE == 0, hgexport(h2,strcat(imgPathFnameFull,'.eps')), end
        saveas(h2,strcat(imgPathFnameFull,'.png'));
        print(h2,'-djpeg',strcat(imgPathFnameFull,'.jpg'));

end


end %End of function
