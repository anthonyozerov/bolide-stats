function quaternion=anglevector2quat(angle,vector)
%
% Converts a vector and the angle to rotate around it into a quaternion
% Assumes a unit vector

quaternion(1) = cos(angle/2) ; 
quaternion(2:4) = vector.*sin(angle/2) ;
quaternion = quaternion_normalize(quaternion) ;% normalize to make sure represents a rotation

return