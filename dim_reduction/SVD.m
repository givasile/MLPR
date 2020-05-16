% Illustration of SVD
% SVD decomposes A into a weighted sum of outter products.
% weights are the singular values S_i
% outter product is between U_i and V_i
% if the sum is exhaustive (i.e. for all i (S_i*U_i*V_i')
% then the initial matrix A is reconstructed perfectly


% Example 1: lets isolate only one element of the diagonal of S
% we show that S_r*U_r*V_r' equals U*S*V' where in S we have zero
% everything apart from the element S(r,r)
r = 1;
A = [1 0 1; -1 -2 0; 0 1 -1];
[U,S,V] = svd(A);
U_r = U(:,r);
V_r = V(:,r);
S_r = S(r,r);
tmp1 = U_r*S_r*V_r';

% zero everything apart from element S(r,r)
[U,S,V] = svd(A);
tmp = S(r,r);
S = zeros(size(S));
S(r,r) = tmp;
tmp2 = U*S*V';

% show that tmp1, tmp2 are equal
sprintf('Example 1: %d', (isequal(tmp1, tmp2)))


% Example 2: show that if we sum over all singular values
% we perfectly reconstruct matrix A
[U,S,V] = svd(A);
A1 = zeros(size(A));
for r = 1:size(S, 1)
    A1 = A1 + U(:,r)*S(r,r)*V(:,r)';
end
sprintf('Example 2: %d', isClose(A,A1))


% Example 3: show that U*S = A*V and U*S*V'=A*V*V' in all cases
% Hence, V is the projection matrix and V' the reprojection
[U,S,V] = svd(A);
sprintf('Example 3: %d', isClose(U*S, A*V))
sprintf('Example 3: %d', isClose(U*S*V', A*V*V'))

S(3,3) = 0;
S(2,2) = 0;
V(:,3) = 0;
V(:,2) = 0;
sprintf('Example 3: %d', isClose(U*S, A*V))
sprintf('Example 3: %d', isClose(U*S*V', A*V*V'))

% Example 4: show that U'*A = S*V and U*S*V'=U*U'*A in all cases
% Hence, V is the projection matrix and V' the reprojection
[U,S,V] = svd(A);
sprintf('Example 4: %d', isClose(U'*A, S*V'))
sprintf('Example 4: %d', isClose(U*S*V', U*U'*A))

S(3,3) = 0;
S(2,2) = 0;
U(:,3) = 0;
U(:,2) = 0;
sprintf('Example 4: %d', isClose(U'*A, S*V'))
sprintf('Example 4: %d', isClose(U*S*V', U*U'*A))


% Example 5: symmetry between column-based/row-based X
% If X = U_1*S_1*V_1' and X' = U_2*S_2*V_2' then
% U_2 = V_1, S_2=S_1', V_2=U_2'
[U_1,S_1,V_1] = svd(A);
[U_2,S_2,V_2] = svd(A');
sprintf('Example 5: %d', isColumnsClose(U_2, V_1))
sprintf('Example 5: %d', isColumnsClose(S_2, S_1'))
sprintf('Example 5: %d', isColumnsClose(V_2, U_1))


function [res] = isClose(A, B)
    
    res = sum(sum(abs(A-B) > 1.0e-10)) == 0;
end

function [res] = isColumnsClose(A, B)
% Checks that all columns of the two input matrices are either the same
% (i.e. A_j==B_j) or opposite (i.e. A_j==-B_j) 

    % set initially true
    res = 1;
    
    % if any is wrong then set to false
    for r = 1:size(A,2)
        if isClose(abs(A(:,r)'*B(:,r)), abs(A(:,r)'*A(:,r))) == 0
            res = 0;
        end
    end
end
