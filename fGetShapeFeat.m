function shapeFeatVector = fGetShapeFeat(region)
% This function gets the shapeFeat descriptors from a binary region
%
% INPUT
%   - region: Image with a binary region to get the descriptors from. There
%   is ONLY ONE REGION IN THE IMAGE. The type of the image must be LOGICAL
%
% OUTPUT
%   shapeFeatVector: Vector with all the descriptors of the ShapeFeat.
%   These are:
%     · Convex Area: Number of pixels of the smallest convex polygon that
%     contains the region.
%     · Eccentricity: Scalar that specifies the eccentricity of the ellipse 
%     that has the same second central moments as the region. 
%     · Perimeter: Number of points in the contour of the region.
%     · Equivalent Diameter: Scalar that specifies the diameter of a circle
%      with the same area as the region.
%     · Extent: Scalar that specifies the ratio of the pixels of the
%     region to the pixels in the bounding box around the region.
%     · Filled Area: Number of pixels belonging to the region after filling
%     its possible holes.
%     · Minor Axis Length: Length of the minor axis of the ellipse that has
%     the same second central moments as the region. 
%     · Major Axis Length: Length of the major axis of the ellipse that has
%     the same second central moments as the region.
%     · R: Ratio between the major and minor axis of the ellipse that has
%     the same second central moments as the region.
%     · Solidity: Ratio between the area of the region and the Convex Area
%     of the region.
%

% Initialise the output
shapeFeatVector = zeros(1,10);

% Look at the Matlab's regionprops function to get the properties of the
% region that you need to get the descriptors of ShapeFeats (mentioned
% above)
% ====================== YOUR CODE HERE ======================

%Convex Area :

%shapeFeatVector(1) = sum(sum(region));

stats = regionprops(region,'Area','Eccentricity','Perimeter','EquivDiameter','Extent','FilledArea','MajorAxisLength','MinorAxisLength','Solidity');

Area = getfield(stats,'Area');
Eccentricity = getfield(stats,'Eccentricity');
Perimeter = getfield(stats,'Perimeter');
EquivDiameter = getfield(stats,'EquivDiameter');
Extent = getfield(stats,'Extent');
FilledArea = getfield(stats,'FilledArea');
MajorAxisLength = getfield(stats,'MajorAxisLength');
MinorAxisLength = getfield(stats,'MinorAxisLength');
Solidity = getfield(stats,'Solidity');

% ============================================================

% Convex Area: Number of pixels of the smallest convex polygon that
% contains the region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(1) = Area;
% ============================================================

% Eccentricity: Scalar that specifies the eccentricity of the ellipse that
% has the same second central moments as the region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(2) = Eccentricity;
% ============================================================

% Perimeter: Number of points in the contour of the region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(3) = Perimeter;
% ============================================================

% Equivalent Diameter: Scalar that specifies the diameter of a circle with
% the same area as the region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(4) = EquivDiameter;
% ============================================================

% Extent: Scalar that specifies the ratio of the pixels of the region to
% the pixels in the bounding box around the region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(5) = Extent;
% ============================================================

% Filled Area: Number of pixels belonging to the region after filling its
% possible holes.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(6) = FilledArea;
% ============================================================

% Minor Axis Length: Length of the minor axis of the ellipse that has the
% same second central moments as the region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(7) = MajorAxisLength;
% ============================================================

% Major Axis Length: Length of the major axis of the ellipse that has the
% same second central moments as the region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(8) = MinorAxisLength;
% ============================================================

% R: Ratio between the major and minor axis of the ellipse that has the
% same second central moments as the region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(9) = MinorAxisLength/MajorAxisLength;
% ============================================================

% Solidity: Ratio between the area of the region and the Convex Area of the
% region.
% ====================== YOUR CODE HERE ======================
shapeFeatVector(10) = Solidity;
% ============================================================


end