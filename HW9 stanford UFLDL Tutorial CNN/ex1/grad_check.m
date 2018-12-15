function average_error = grad_check(fun, theta0, num_checks, varargin)

  delta=1e-3; 
  sum_error=0;

  fprintf(' Iter       i             err');
  fprintf('           g_est               g               f\n')
    num_checks = 20;
  for i=1:num_checks
    %T = theta0;
    T = theta;
    j = randsample(numel(T),1);
    T0=T; T0(j) = T0(j)-delta;
    T1=T; T1(j) = T1(j)+delta;

%     [f,g] = fun(T, varargin{:});
%     f0 = fun(T0, varargin{:});
%     f1 = fun(T1, varargin{:});
     [f,g] = logistic_regression_vec(T, train.X, train.y);
     [f0,g0] = logistic_regression_vec(T0, train.X, train.y);
     [f1,g1] = logistic_regression_vec(T1, train.X, train.y);
    g_est = (f1-f0) / (2*delta);
    error = abs(g(j) - g_est);

    fprintf('% 5d  % 6d % 15g % 15f % 15f % 15f\n', ...
            i,j,error,g(j),g_est,f);

    sum_error = sum_error + error;  
  end

  average=sum_error/num_checks;
