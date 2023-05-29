function quaternion=vec_vec2quat(v1,v2)
%
% Find quaternion that describes rotation from one vector to another
%
% Angle between 2 vectors is acos(v1.v2)  (assuming unit vectors)
% Rotation axis is v1 x v2

angle = acos(dot(v1,v2)) ; 
vector = cross(v1,v2) ; 
% normalize vector if possible
magv = sqrt(sum(vector.^2)) ;
if magv
    vector = vector/magv ; 
end
quaternion=anglevector2quat(angle,vector) ; 

return