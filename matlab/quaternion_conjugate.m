function qstar = quaternion_conjugate(q)
% invert quaternion

qstar=q ; 
qstar(2:4)=-qstar(2:4);

end