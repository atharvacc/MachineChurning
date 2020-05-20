%TODO: append the image name to csv file
clear;
setNumber = '1';
%Read in MUSE and H&E images
MUSEimagepath = '/Users/moni/Desktop/images/Stack0000_real_A.png';
HEimagepath = '/Users/moni/Desktop/images/Stack0000_fake_B.png';
[MUSEOriginalImage] = imread(MUSEimagepath);
[HEOriginalImage] = imread(HEimagepath);

museImgName = PathNameParser(MUSEimagepath);
heImgName = PathNameParser(HEimagepath);

%disp( museImgName);
%disp( heImgName);

%Convert both to gray scale
MUSEGrayScaled = rgb2gray(MUSEOriginalImage);
HEGrayScaled = rgb2gray(HEOriginalImage);

%display the original images

figure;
subplot(1,2,1);
imshow(MUSEOriginalImage);
title('MUSE: Original Image');

subplot(1,2,2);
imshow(HEOriginalImage);
title('H&E: Original Image');

%display gray scaled images
figure;
ax1 = subplot(1,2,1);
imshow(MUSEGrayScaled);
title('MUSE: Gray Scaled');
hold on;

ax2 = subplot(1,2,2);
imshow(HEGrayScaled);
title('H&E: Gray Scaled');
hold on;

%select points from gray scaled images

npoints = 30; %number of points being selected
MUSEcoordinates= zeros(npoints,2);
HEcoordinates = zeros(npoints,2);
color = {'b', 'g', 'r', 'c', 'm', 'y'};
symbol = {'.','o','x','+', '*', 's', 'd', 'v'};

colorIndex = 0;
maxColors = 6;
symbolIndex = 0;
maxSymbols = 8;


for idx = 1:npoints
    fprintf('Selecting point %d\n', idx);

    colorIndex = colorIndex +1;
    symbolIndex = symbolIndex +1;

    currMarker = append(color(colorIndex),symbol(symbolIndex));

    [X1,Y1] = getpts(ax1);
    plot(X1,Y1, string(currMarker));
    fprintf('Muse point %d Value %d\n',idx, MUSEGrayScaled(floor(Y1),floor(X1)));
    MUSEcoordinates(idx,1) = idx;
    MUSEcoordinates(idx,2) = MUSEGrayScaled(floor(Y1),floor(X1));

    [X2,Y2] = getpts(ax2);
    plot(X2,Y2, string(currMarker));
    fprintf('HE point %d Value %d\n', idx, HEGrayScaled(floor(Y2),floor(X2)));
    HEcoordinates(idx,1) = idx;
    HEcoordinates(idx,2) = HEGrayScaled(floor(Y2),floor(X2));

    if(colorIndex == maxColors)
        colorIndex = 0;
    end

    if(symbolIndex == maxSymbols)
        symbolIndex = 0;
    end

end

%disp(museImage);
%disp(heImage);

museImage = append('selectedMUSEpnts',museImgName,'_',setNumber,'.csv');
heImage = append('selectedHEpnts',heImgName,'_',setNumber,'.csv');

writematrix(MUSEcoordinates,string(museImage));
writematrix(HEcoordinates,string(heImage));
