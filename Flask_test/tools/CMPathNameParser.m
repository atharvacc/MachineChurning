function [imageName] = CMPathNameParser(imgPathName)
%CMPathNameParser obtaining the stack number of the image from imgPathName
%   Obtain the Stack number of image
%   Given passed in image is in the following format: Stackxxxx_y_z.png

newstr = split(imgPathName,["_", "/"]);

strsize = size(newstr);
 for index = 1:strsize(1)
     if (contains(newstr(index),"Stack"))
         imageName = newstr(index);
     end
 end

end
