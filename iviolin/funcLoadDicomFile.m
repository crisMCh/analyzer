function [FName, Label, ROI1, ROI2] = funcLoadDicomFile(PatientNumber)
  
switch PatientNumber
% Good cases
  case 10
    FName = 'Pat10-I0000161';
    Label = 1;
    ROI1 = [327,336]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [313,338]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND
  case 11
    FName = 'Pat11-I0000147'
    Label = 1;
    ROI1 = [351,322]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [392,332]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND
  case 12
    FName = 'Pat12-I0000148'
    Label = 1;
    ROI1 = [367,297]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [389,286]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 19
    FName = 'Pat19-I0000119'
    Label = 1;
    ROI1 = [317,348]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [313,338]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 20
    FName = 'Pat20-I0000198'
    Label = 1;
    ROI1 = [417,280]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [421,305]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 21
    FName = 'Pat21-I0000213'
    Label = 1;
    ROI1 = [338,272]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [375,238]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 22
    FName = 'Pat22-I0000198'
    Label = 1;
    ROI1 = [417,280]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [421,305]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 60
    FName = 'Pat60-12072689_I0000139'
    Label = 1;
    ROI1 = [393,297]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [389,307]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 61
    FName = 'Pat61-12271388_I0000150'
    Label = 1;
    ROI1 = [331,304]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [320,314]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 62
    FName = 'Pat62-12287621_I0000114'
    Label = 1;
    ROI1 = [354,285]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [369,296]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 63
    FName = 'Pat63-12346605_I0000180'
    Label = 1;
    ROI1 = [373,278]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [369,288]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 64
    FName = 'Pat64-12374341_I0000180'
    Label = 1;
    ROI1 = [390,309]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [411,283]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 65
    FName = 'Pat65-12641614_I0000230'
    Label = 1;
    ROI1 = [368,303]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [357,313]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 66
    FName = 'Pat66-12714933_I0000172'
    Label = 1;
    ROI1 = [376,276]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [385,292]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 67
    FName = 'Pat67-12776322_I0000150'
    Label = 1;
    ROI1 = [332,308]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [400,290]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
  case 68
    FName = 'Pat68-12827646_I0000180'
    Label = 1;
    ROI1 = [346,307]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [366,300]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND    
% Bad cases    
  case 13
    FName = 'Pat13-IM0107'
    Label = 0;
    ROI1 = [327,318 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [365,280 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND        
  case 16
    FName = 'Pat16-I0000157'
    Label = 0;
    ROI1 = [356,321 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [361,335 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 18
    FName = 'Pat18-I0000175'
    Label = 0;
    ROI1 = [348,347 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [313,338 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 48
    FName = 'Pat48-I0000125'
    Label = 0;
    ROI1 = [356,340 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [342,309 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 49
    FName = 'Pat49-I0000192'
    Label = 0;
    ROI1 = [381,296 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [383,304 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 50
    FName = 'Pat50-12295142_I0000290'
    Label = 0;
    ROI1 = [409,264 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [407,272 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 51
    FName = 'Pat51-12311511_I0000113'
    Label = 0;
    ROI1 = [329,343 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [379,326 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 52
    FName = 'Pat52-12557264_I0000160'
    Label = 0;
    ROI1 = [378,307 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [395,302 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 53
    FName = 'Pat53-12591011_I0000190'
    Label = 0;
    ROI1 = [402,307 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [392,326 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 54
    FName = 'Pat54-12623833_I0000146'
    Label = 0;
    ROI1 = [375,286 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [370,296 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 55
    FName = 'Pat55-12663282_I0000250'
    Label = 0;
    ROI1 = [371,305 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [380,333 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 56
    FName = 'Pat56-12688563_I0000200'
    Label = 0;
    ROI1 = [387,342 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [399,288 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
  case 57
    FName = 'Pat57-20180330_IM0039'
    Label = 0;
    ROI1 = [354,308 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [342,329 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
case 58
    FName = 'Pat58-20180509_IM0082'
    Label = 0;
    ROI1 = [384,341 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [356,342 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND   
case 59
    FName = 'Pat59-I0000221'
    Label = 0;
    ROI1 = [357,315 ]; %ROI 1 coordinates [X,Y,ROI_SIZE] (MAJOR FISSURE)
    ROI2 = [370,309 ]; %ROI 2 coordinates [X,Y,ROI_SIZE] (BACKGROUND                  
end