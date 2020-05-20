function [imageName] = PathNameParser(imagePath)
%Path name parser
%   Looking for image name in the pathname
newstr = split(imagePath,["/","."]);

strsize = size(newstr);
 for index = 1:strsize(1)
     if (contains(newstr(index),"Stack"))
         imageName = newstr(index);
     end
 end
end
