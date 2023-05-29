function q=quaternion_normalize(q)

q=q./sqrt(sum(q.^2)) ; 

return