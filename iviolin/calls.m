function calls()
    % Define the save path
    save_path = 'G:/Cristina/Thesis/analyzer/figs/iviolin/octave/';

    orig_path = 'G:\Cristina\Thesis\analyzer\predictions\iviolin\original\dcm';
    pred_paths ={'G:\Cristina\Thesis\analyzer\predictions\iviolin\WIT_e105_n40000\dcm', 'G:\Cristina\Thesis\analyzer\predictions\iviolin\WIT_e105_n20000\dcm', 'G:\Cristina\Thesis\analyzer\predictions\iviolin\WIT_e155_n10000\dcm'};

    % Define file names
    slicesi2 = {'prediction_0000.dcm', 'prediction_0002.dcm', 'prediction_0004.dcm', 'prediction_0006.dcm'};
    originalsi2 = {'iviolin_abdo_pat1_target.dcm', 'iviolin_thor_acc_pat1_target.dcm', 'iviolin_thor_acc_pat3_target.dcm', 'iviolin_thor_nacc_pat2_target.dcm'};
    roisi1s = [158, 382, 18, 18;  327, 336, 18, 18;  367, 297, 18, 18; 356, 321, 18, 18];
    roisi2s = [142, 369, 18, 18;  313, 338, 18, 18; 389, 286, 18, 18;  325, 363, 18, 18 ];


    #iviolin
    slicesi = {"prediction_0000.dcm","prediction_0001.dcm","prediction_0002.dcm", "prediction_0003.dcm", "prediction_0004.dcm", "prediction_0005.dcm", "prediction_0006.dcm", "prediction_0007.dcm", "prediction_0008.dcm"}
    originalsi = {"iviolin_abdo_pat1_target.dcm","iviolin_abdo_pat2_target.dcm","iviolin_thor_acc_pat1_target.dcm", "iviolin_thor_acc_pat2_target.dcm", "iviolin_thor_acc_pat3_target.dcm", "iviolin_thor_acc_pat4_target.dcm", "iviolin_thor_nacc_pat2_target.dcm", "iviolin_thor_nacc_pat3_target.dcm", "iviolin_thor_nacc_pat4_target.dcm"}
    

    % Define ROIs
   %                 0               1               2                   3               4                   5                   6               7              8        
    roisi1 = [158, 382, 18, 18;  315,289,18,18; 327, 336, 18, 18; 351, 322  ,18,18; 367, 297, 18, 18; 317, 348 ,18,18; 356, 321, 18, 18; 409,264 ,18,18; 329, 343,18,18];
    roisi2 = [142, 369, 18, 18;  330,296,18,18; 313, 338, 18, 18;  392, 332,18,18; 389, 286, 18, 18; 313, 338 ,18,18; 325, 363, 18, 18; 407, 272,18,18; 379,326,18,18];

    for i = 1:length(originalsi)
        % Construct file path for prediction
        orig_file = fullfile(orig_path, originalsi{i});
        pred_file_40k = fullfile(pred_paths{1}, slicesi{i});
        pred_file_20k = fullfile(pred_paths{2}, slicesi{i});
        pred_file_10k = fullfile(pred_paths{3}, slicesi{i});

        % Call the Octave function for prediction and save the values in their own lists
        [f_int{1}, MTF_int{1}, fNPS1{1}, NPS1_int{1}, fNPS2{1}, NPS2{1}, fMTF{1}, cont_ROI1{1}, cont_ROI2{1}, MTF_area{1}, PS_area{1}, NPS_area{1}, IMGreturn{1}, Manufacturer{1}, Model{1}, ExposureTime{1}, TubeCurrent{1}, WindowWidth{1}, WindowCenter{1}, PredictedLabel{1}, DecisionValue{1}] = ...
            funcImageQualityOctave(pred_file_40k, roisi1(i,3), [roisi1(i,1), roisi1(i,2)], [roisi2(i,1), roisi2(i,2)]);

        [f_int{2}, MTF_int{2}, fNPS1{2}, NPS1_int{2}, fNPS2{2}, NPS2{2}, fMTF{2}, cont_ROI1{2}, cont_ROI2{2}, MTF_area{2}, PS_area{2}, NPS_area{2}, IMGreturn{2}, Manufacturer{2}, Model{2}, ExposureTime{2}, TubeCurrent{2}, WindowWidth{2}, WindowCenter{2}, PredictedLabel{2}, DecisionValue{2}] = ...
            funcImageQualityOctave(pred_file_20k, roisi1(i,3), [roisi1(i,1), roisi1(i,2)], [roisi2(i,1), roisi2(i,2)]);

        [f_int{3}, MTF_int{3}, fNPS1{3}, NPS1_int{3}, fNPS2{3}, NPS2{3}, fMTF{3}, cont_ROI1{3}, cont_ROI2{3}, MTF_area{3}, PS_area{3}, NPS_area{3}, IMGreturn{3}, Manufacturer{3}, Model{3}, ExposureTime{3}, TubeCurrent{3}, WindowWidth{3}, WindowCenter{3}, PredictedLabel{3}, DecisionValue{3}] = ...
            funcImageQualityOctave(pred_file_10k, roisi1(i,3), [roisi1(i,1), roisi1(i,2)], [roisi2(i,1), roisi2(i,2)]);

        [f_int_orig, MTF_int_orig, fNPS1_orig, NPS1_int_orig, fNPS2_orig, NPS2_orig, fMTF_orig , cont_ROI1_orig, cont_ROI2_orig, MTF_area_orig, PS_area_orig, NPS_area_orig, IMGreturn_orig, Manufacturer_orig, Model_orig, ExposureTime_orig, TubeCurrent_orig, WindowWidth_orig, WindowCenter_orig, PredictedLabel_orig, DecisionValue_orig] = ...
            funcImageQualityOctave(orig_file, roisi1(i,3), [roisi1(i,1), roisi1(i,2)], [roisi2(i,1), roisi2(i,2)]);

        plot_MTF(MTF_int, MTF_int_orig, save_path, originalsi{i});
        plot_NPS(NPS2, NPS2_orig, save_path, originalsi{i});
        plot_PS(NPS1_int, NPS1_int_orig,save_path, originalsi{i});

    end

    %for i = 1:length(originalsi2)
    %    orig_file = fullfile(orig_path, originalsi2{i});
%
    %    [f_int_orig, MTF_int_orig, fNPS1_orig, NPS1_int_orig, fNPS2_orig, NPS2_orig, fMTF_orig , cont_ROI1_orig, cont_ROI2_orig, MTF_area_orig, PS_area_orig, NPS_area_orig, IMGreturn_orig, Manufacturer_orig, Model_orig, ExposureTime_orig, TubeCurrent_orig, WindowWidth_orig, WindowCenter_orig, PredictedLabel_orig, DecisionValue_orig] = ...
    %        funcImageQualityOctave(orig_file, roisi1s(i,3), [roisi1s(i,1), roisi1s(i,2)], [roisi2s(i,1), roisi2s(i,2)]);
%
%
    %    plot_MTF(MTF_int, MTF_int_orig, save_path, originalsi2{i});
    %    plot_NPS(NPS2, NPS2_orig, save_path, originalsi2{i});
    %    plot_PS(NPS1_int, NPS1_int_orig,save_path, originalsi2{i});
%
    %end




end

function plot_MTF(MTF1_list, MTF2, save_path, filename)
    % Plot all MTF1 in the list and MTF2 in one graph
    figure('Color', 'w'); % Set the figure background color to white
    hold on;
    
    % Define a colormap to ensure different colors for each plot
    colors = lines(length(MTF1_list));
    
    % Plot each MTF1 in the list with unique colors
    for i = 1:length(MTF1_list)
        plot(MTF1_list{i}, 'DisplayName', ['Noise level ', num2str(i)], 'LineWidth', 2, 'Color', colors(i, :));
    end
    
    % Plot MTF2 with black color and thicker line
    plot(MTF2, 'DisplayName', 'Original', 'LineWidth', 3, 'Color', 'k');
    
    title('Modulation Transfer Function (MTF)');
    xlabel('Spatial Frequency [mm^{-1}]');
    xticks = get(gca, 'XTick');
    set(gca, 'XTickLabel', xticks / 100);
    ylabel('MTF');
    ylim([0 1]);
    grid on;
    legend('show');
    
    % Save the figure
    saveas(gcf, fullfile(save_path, [filename, '_MTFs.png']));
    #waitfor(gcf);
    close(gcf);
end

function plot_NPS(NPS1_list, NPS2, save_path, filename)
    % Ensure NPS2 matches the length of NPS1
    NPS2_resampled = interp1(linspace(0,1,length(NPS2)), NPS2, linspace(0,1,length(NPS1_list{1})), 'linear');

    % Plot all NPS1 in the list and resampled NPS2 in one graph
    figure('Color', 'w'); % Set the figure background color to white
    hold on;
    
    % Define a colormap to ensure different colors for each plot
    colors = lines(length(NPS1_list));
    
    % Plot each NPS1 in the list with unique colors
    for i = 1:length(NPS1_list)
        plot(NPS1_list{i}, 'DisplayName', ['Noise level ', num2str(i)], 'LineWidth', 2, 'Color', colors(i, :));
    end
    
    % Plot resampled NPS2 with black color and thicker line
    plot(NPS2_resampled, 'DisplayName', 'Original', 'LineWidth', 3, 'Color', 'k');
    
    title('Noise Power Spectrum (NPS)');
    xlabel('Spatial Frequency [mm^{-1}]');
    xticks = get(gca, 'XTick');
    set(gca, 'XTickLabel', xticks / 100);
    ylabel('NPS');
    grid on;
    legend('show');
    
    % Save the figure
    saveas(gcf, fullfile(save_path, [filename, '_NPSs.png']));
    %waitfor(gcf);
    close(gcf);
end

function plot_PS(NPS1_list, NPS2, save_path, filename)
    % Ensure NPS2 matches the length of NPS1
    NPS2_resampled = interp1(linspace(0,1,length(NPS2)), NPS2, linspace(0,1,length(NPS1_list{1})), 'linear');

    % Plot all NPS1 in the list and resampled NPS2 in one graph
    figure('Color', 'w'); % Set the figure background color to white
    hold on;
    
    % Define a colormap to ensure different colors for each plot
    colors = lines(length(NPS1_list));
    
    % Plot each NPS1 in the list with unique colors
    for i = 1:length(NPS1_list)
        plot(NPS1_list{i}, 'DisplayName', ['Noise level ', num2str(i)], 'LineWidth', 2, 'Color', colors(i, :));
    end
    
    % Plot resampled NPS2 with black color and thicker line
    plot(NPS2_resampled, 'DisplayName', 'Original', 'LineWidth', 3, 'Color', 'k');
    
    title('Power Spectrum (PS)');
    xlabel('Spatial Frequency [mm^{-1}]');
    xticks = get(gca, 'XTick');
    set(gca, 'XTickLabel', xticks / 100);
    ylabel('NPS');
    grid on;
    legend('show');
    
    % Save the figure
    saveas(gcf, fullfile(save_path, [filename, '_PSs.png']));
    %waitfor(gcf);
    close(gcf);
end

function plot_PS_MTF(fMTF1, MTF1, fNPS1, NPS1, fMTF2, MTF2, fNPS2, NPS2, save_path, filename)
%DOESN' WORK
    % Preprocess data for prediction
    fMTF1 = fMTF1(2:end);
    MTF1 = MTF1(2:end);
    fMin1 = fMTF1(1);
    fMax1 = fNPS1(end);
    f_int1 = fMin1:0.02:fMax1;
    MTF_int1 = interp1(fMTF1, MTF1, f_int1);
    NPS_int1 = interp1(fNPS1, NPS1, f_int1);

    % Preprocess data for original
    fMTF2 = fMTF2(2:end);
    MTF2 = MTF2(2:end);
    fMin2 = fMTF2(1);
    fMax2 = fNPS2(end);
    f_int2 = fMin2:0.02:fMax2;
    MTF_int2 = interp1(fMTF2, MTF2, f_int2);
    NPS_int2 = interp1(fNPS2, NPS2, f_int2);


    % Generate and adjust plot (not visible)
    h = figure('Position', [100 100 700 700], "visible", "off");
    hold on;
    grid on;

    % Plot prediction data
    plot(f_int1, NPS_int1 ./ MTF_int1, 'DisplayName', 'Prediction', 'LineWidth', 2);

    % Plot original data
    plot(f_int2, NPS_int2 ./ MTF_int2, 'DisplayName', 'Original', 'LineWidth', 2);

    xlabel('Spatial frequency [mm^{-1}]');
    ylabel('Intensity [a.u.]');
    xlim([min(fMin1, fMin2) max(fMax1, fMax2)]);
    legend('show');

    % Save the figure
    saveas(h, fullfile(save_path, [filename, '_PStoMTF.png']));
    close(h);
end




% Call the main function
calls();