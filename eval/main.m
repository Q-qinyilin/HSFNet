%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluation tool boxs for PraNet: Parallel Reverse Attention Network for Polyp Segmentation (MICCAI20).
%Author: Deng-Ping Fan, Tao Zhou, Ge-Peng Ji, Yi Zhou, Geng Chen, Huazhu Fu, Jianbing Shen, and Ling Shao
%Homepage: http://dpfan.net/
%Projectpage: https://github.com/DengPingFan/PraNet
%First version: 2020-6-28
%Any questions please contact with dengpfan@gmail.com.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Function: Providing several important metrics: Dice, IoU, F1, S-m (ICCV'17), Weighted-F1 (CVPR'14)
%          E-m (IJCAI'18), Precision, Recall, Sensitivity, Specificity, MAE.


clear all;
close all;
clc;

% ---- 1. ResultMap Path Setting ----
ResultMapPath = 'C:/Users/qin/Desktop/HSF-Net/';
Models = {'HSFNet'}; %{'UNet','UNet++','PraNet','SFA'};
modelNum = length(Models);

% ---- 2. Ground-truth Datasets Setting ----
DataPath = 'E:/dataset/Polyp_TestDataset/'; % E:\dataset\Polyp_TestDataset
Datasets = {'CVC-ClinicDB', 'CVC-ColonDB','ETIS-LaribPolypDB', 'Kvasir','CVC-300'};

% ---- 3. Evaluation Results Save Path Setting ----
ResDir = './EvaluateResults/';
ResName='_result.txt';  % You can change the result name.

Thresholds = 1:-1/255:0;
dice_iou_threshold = 0.5; 
datasetNum = length(Datasets);

fprintf('=== Path Debugging ===\n');
fprintf('ResultMapPath: %s (exists: %d)\n', ResultMapPath, exist(ResultMapPath, 'dir'));
fprintf('DataPath: %s (exists: %d)\n', DataPath, exist(DataPath, 'dir'));
fprintf('ResDir: %s (exists: %d)\n', ResDir, exist(ResDir, 'dir'));

for d = 1:datasetNum
    
    tic;
    dataset = Datasets{d}   % print cur dataset name
    fprintf('Processing %d/%d: %s Dataset\n',d,datasetNum,dataset);
    
    gtPath = [DataPath dataset '/masks/'];
    fprintf('Ground truth path: %s (exists: %d)\n', gtPath, exist(gtPath, 'dir'));
    
    ResPath = [ResDir dataset '-mat/']; % The result will be saved in *.mat file so that you can used it for the next time.
    if ~exist(ResPath,'dir')
        mkdir(ResPath);
    end
    resTxt = [ResDir dataset ResName];  % The evaluation result will be saved in `../Resluts/Result-XXXX` folder.
    fileID = fopen(resTxt,'w');
    
    if fileID == -1
        fprintf('Error: Cannot create result file: %s\n', resTxt);
        continue;
    end
    
    for m = 1:modelNum
        model = Models{m}   % print cur model name
        
        resMapPath = [ResultMapPath '/' model '/' dataset '/'];
        
        fprintf('Result map path: %s (exists: %d)\n', resMapPath, exist(resMapPath, 'dir'));
        
        imgFiles = dir([resMapPath '*.png']);
        imgNUM = length(imgFiles);
        
        fprintf('Found %d images in result path\n', imgNUM);
        
        if imgNUM == 0
            fprintf('Warning: No PNG files found in %s\n', resMapPath);
            fprintf('Checking if directory exists and listing contents:\n');
            if exist(resMapPath, 'dir')
                allFiles = dir(resMapPath);
                for f = 1:length(allFiles)
                    if ~allFiles(f).isdir
                        fprintf('  Found file: %s\n', allFiles(f).name);
                    end
                end
            else
                fprintf('  Directory does not exist!\n');
            end
            continue;
        end
        
        % Keep original arrays for other metrics
        [threshold_Fmeasure, threshold_Emeasure] = deal(zeros(imgNUM,length(Thresholds)));
        [threshold_Precion, threshold_Recall] = deal(zeros(imgNUM,length(Thresholds)));
        [threshold_Sensitivity, threshold_Specificity] = deal(zeros(imgNUM,length(Thresholds)));
        
        % Add separate arrays for single-threshold Dice and IoU
        [dice_values, iou_values] = deal(zeros(1,imgNUM));
        [Smeasure, wFmeasure, MAE] =deal(zeros(1,imgNUM));
        
        for i = 1:imgNUM
            name =  imgFiles(i).name;
            fprintf('Evaluating(%s Dataset,%s Model, %s Image): %d/%d\n',dataset, model, name, i,imgNUM);
            
            %load gt
            gtFile = [gtPath name];
            if ~exist(gtFile, 'file')
                fprintf('Warning: Ground truth file not found: %s\n', gtFile);
                continue;
            end
            gt = imread(gtFile);
            
            if (ndims(gt)>2)
                gt = rgb2gray(gt);
            end
            
            if ~islogical(gt)
                gt = gt(:,:,1) > 128;
            end
            
            %load resMap
            resFile = [resMapPath name];
            if ~exist(resFile, 'file')
                fprintf('Warning: Result file not found: %s\n', resFile);
                continue;
            end
            resmap  = imread(resFile);
            %check size
            if size(resmap, 1) ~= size(gt, 1) || size(resmap, 2) ~= size(gt, 2)
                resmap = imresize(resmap,size(gt));
                imwrite(resmap,[resMapPath name]);
                fprintf('Resizing have been operated!! The resmap size is not math with gt in the path: %s!!!\n', [resMapPath name]);
            end
            
            resmap = im2double(resmap(:,:,1));
            
            resmap = reshape(mapminmax(resmap(:)',0,1),size(resmap));
            
            Smeasure(i) = StructureMeasure(resmap,logical(gt));
            
            wFmeasure(i) = original_WFb(resmap,logical(gt));
            
            MAE(i) = mean2(abs(double(logical(gt)) - resmap));
            
            gt_binary = logical(gt);
            pred_binary = resmap >= dice_iou_threshold;
            
            intersection = sum(sum(gt_binary & pred_binary));
            union = sum(sum(gt_binary | pred_binary));
            
            epsilon = 1e-6;
            dice_values(i) = (2 * intersection + epsilon) / (sum(sum(gt_binary)) + sum(sum(pred_binary)) + epsilon);
            iou_values(i) = (intersection + epsilon) / (union + epsilon);
            
            [threshold_E, threshold_F, threshold_Pr, threshold_Rec]  = deal(zeros(1,length(Thresholds)));
            [threshold_Spe]  = deal(zeros(1,length(Thresholds)));
            for t = 1:length(Thresholds)
                threshold = Thresholds(t);
                [threshold_Pr(t), threshold_Rec(t), threshold_Spe(t), ~, threshold_F(t), ~] = Fmeasure_calu(resmap,double(gt),size(gt),threshold);
                
                Bi_resmap = zeros(size(resmap));
                Bi_resmap(resmap>=threshold)=1;
                threshold_E(t) = Enhancedmeasure(Bi_resmap, gt);
            end
            
            threshold_Emeasure(i,:) = threshold_E;
            threshold_Fmeasure(i,:) = threshold_F;
            threshold_Sensitivity(i,:) = threshold_Rec;
            threshold_Specificity(i,:) = threshold_Spe;
            
        end
        
        %MAE
        mae = mean2(MAE);
        
        %Sm
        Sm = mean2(Smeasure);
        
        %wFm
        wFm = mean2(wFmeasure);
        
        %E-m (keep original calculation)
        column_E = mean(threshold_Emeasure,1);
        meanEm = mean(column_E);
        maxEm = max(column_E);
        
        %Sensitivity (keep original calculation)
        column_Sen = mean(threshold_Sensitivity,1);
        meanSen = mean(column_Sen);
        maxSen = max(column_Sen);
        
        %Specificity (keep original calculation)
        column_Spe = mean(threshold_Specificity,1);
        meanSpe = mean(column_Spe);
        maxSpe = max(column_Spe);
        
        % Modified: Use single-threshold Dice and IoU
        meanDic = mean(dice_values);
        meanIoU = mean(iou_values);
        maxDic = meanDic;  % Since we only have one threshold
        maxIoU = meanIoU;  % Since we only have one threshold
        
        save([ResPath model],'Sm', 'mae', 'dice_values', 'column_Sen', 'column_Spe', 'column_E','iou_values','maxDic','maxEm','maxSen','maxSpe','maxIoU','meanIoU','meanDic','meanEm','meanSen','meanSpe');
        fprintf(fileID, '(Dataset:%s; Model:%s) meanDic:%.3f;meanIoU:%.3f;wFm:%.3f;Sm:%.3f;meanEm:%.3f;MAE:%.3f;maxEm:%.3f;maxDice:%.3f;maxIoU:%.3f;meanSen:%.3f;maxSen:%.3f;meanSpe:%.3f;maxSpe:%.3f.\n',dataset,model,meanDic,meanIoU,wFm,Sm,meanEm,mae,maxEm,maxDic,maxIoU,meanSen,maxSen,meanSpe,maxSpe);
        fprintf('(Dataset:%s; Model:%s) meanDic:%.3f;meanIoU:%.3f;wFm:%.3f;Sm:%.3f;meanEm:%.3f;MAE:%.3f;maxEm:%.3f;maxDice:%.3f;maxIoU:%.3f;meanSen:%.3f;maxSen:%.3f;meanSpe:%.3f;maxSpe:%.3f.\n',dataset,model,meanDic,meanIoU,wFm,Sm,meanEm,mae,maxEm,maxDic,maxIoU,meanSen,maxSen,meanSpe,maxSpe);
    end
    
    fclose(fileID);
    toc;
end




