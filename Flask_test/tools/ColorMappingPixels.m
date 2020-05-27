clear;
%Read in MUSE and H&E images
MUSEimagepath = '/Users/moni/Desktop/images/Stack0028_real_A.png';
HEimagepath = '/Users/moni/Desktop/images/Stack0028_fake_B.png';
[MUSEOriginalImage] = imread(MUSEimagepath);
[HEImage] = imread(HEimagepath);

stackNum = CMPathNameParser(MUSEimagepath);

%disp( museImgName);
%disp( heImgName);

%display images
figure;
ax1 = subplot(1,2,1);
imshow(MUSEOriginalImage);
title('MUSE: Original Image')
hold on;

ax2 = subplot(1,2,2);
imshow(HEImage);
title('H&E: Output Image');
hold on;

%select points from gray scaled images

npoints = 25; %number of points being selected

%MUSE_R = MUSEOriginalImage(:,:,1);
%MUSE_G = MUSEOriginalImage(:,:,2);
%MUSE_B = MUSEOriginalImage(:,:,3);
%HEcoordinates = zeros(1,npoints);
%HE_R = HEImage(:,:,1);
%HE_G = HEImage(:,:,2);
%HE_B = HEImage(:,:,3);

museCordCol = zeros(1,npoints);
museCordRow = zeros(1,npoints);
heCordCol = zeros(1,npoints);
heCordRow = zeros(1,npoints);

color = {'b', 'g', 'r', 'c', 'm', 'y'};
symbol = {'.','o','x','+', '*', 's', 'd', 'v'};

colorIndex = 0;
maxColors = 6;
symbolIndex = 0;
maxSymbols = 8;


for idx = 1:npoints
    %fprintf('Selecting point %d\n', idx);

    colorIndex = colorIndex +1;
    symbolIndex = symbolIndex +1;

    currMarker = append(color(colorIndex),symbol(symbolIndex));

    [X1,Y1] = getpts(ax1);
    plot(X1,Y1, string(currMarker));
    %fprintf('Muse point %d Value %d\n',idx, MUSEOriginalImage(floor(Y1),floor(X1)));
    %MUSEcoordinates(idx) = [MUSE_R(floor(Y1),floor(X1)), MUSE_G(floor(Y1),floor(X1)), MUSE_B(floor(Y1),floor(X1))];
    museCordRow(idx) = X1;
    museCordCol(idx) = Y1;

    [X2,Y2] = getpts(ax2);
    plot(X2,Y2, string(currMarker));
    %fprintf('HE point %d Value %d\n', idx, HEResultingImage(floor(Y2),floor(X2)));
    %HEcoordinates(idx) = [HE_R(floor(Y1),floor(X1)), HE_G(floor(Y1),floor(X1)), HE_B(floor(Y1),floor(X1))];
    heCordRow(idx) = X2;
    heCordCol(idx) = Y2;

    if(colorIndex == maxColors)
        colorIndex = 0;
    end

    if(symbolIndex == maxSymbols)
        symbolIndex = 0;
    end

end

%disp(museImage);
%disp(heImage);
MUSErgb = impixel(MUSEOriginalImage, museCordCol, museCordRow);
HErgb = impixel(HEImage, heCordCol, heCordRow);

selectedPointDoc = append(stackNum,'_','selectedPnts','.csv');
fileName = fullfile(pwd, string(selectedPointDoc));
%open file
fid = fopen(fileName, 'wt');
%writing headers
fprintf(fid, 'Muse_R, Muse_G, Muse_B, H&E_R, H&_G, H&E_B\n');
%writing data
%fprintf(fid, '%f, %f\n',MUSEcoordinates, HEcoordinates);
fprintf(fid, '%f %f %f, %f %f %f\n', MUSErgb, HErgb);
%close file
fclose(fid);
