# small scrip to compute the solution of Lipman Schinwer equation
# In this case we use the donwsampled function using linear interpolation

using Images
using Plots
using IterativeSolvers
using SpecialFunctions
using FFTW

include("../src/FastConvolutionDual.jl")
include("../src/FastConvolution.jl")
include("../src/Preconditioner.jl")
include("../src/FastConvolutionSlowDual.jl")
include("../src/FastConvolutionDualDownSampled.jl")


FFTW.set_num_threads(length(Sys.cpu_info()))
BLAS.set_num_threads(length(Sys.cpu_info()))

#Defining Omega
h = 0.00025
k = 1/h

# size of box
a = 1;
b = 0.125;
x = collect(-a/2:h:a/2)
y = collect(-b/2:h:b/2)
(n,m) = length(x), length(y)
N = n*m
X = repeat(x, 1, m)[:];
Y = repeat(y', n,1)[:];
# we solve \triangle u + k^2(1 + nu(x))u = 0

# We use the modified quadrature in Duan and Rohklin
(ppw,D) = referenceValsTrapRule();
D0 = D[1];

window(y,alpha) = 1*(abs.(y).<=alpha/2) + (abs.(y).>alpha/2).*(abs.(y).<alpha).*
                     exp.(2*exp.(-0.5*alpha./(abs.(y)-alpha/2))./
                         ((abs.(y).-alpha/2)./(0.5*alpha).-1) )

window(y,alpha, beta) = 1*(abs.(y).<=beta) + (abs.(y).>beta).*(abs.(y).<alpha).*
                           exp.(2.0*exp.(-(alpha- beta)./(abs.(y).-beta))./
                               ((abs.(y).-beta)./(alpha- beta).-1) )

# Defining the smooth perturbation of the slowness
nu(x,y) = -0.05*(sin.(4*pi.*x./(0.96))).*
          window(y,0.96*b/2, 0.48*b/2).*
          window(x,0.96*0.5, 0.3);
          
plot(Gray.(real(reshape( 1 .+  nu(X,Y),n,m))))



fastconvSlowDual = buildFastConvolutionSlowDual(x,y,h,k,nu, [1.0, 0.0],
                                                quadRule ="Greengard_Vico");

rhsSlowDual = -k^2*nu(X,Y) + zeros(ComplexF64,N);

# allocating the solution
sigmaSlow = zeros(ComplexF64,N);

# solving the system using GMRES
@time info =  gmres!(sigmaSlow, fastconvSlowDual, rhsSlowDual, maxiter = 10 )
# println(info[2].residuals[:])

using Interpolations



downsampleX = 16
downsampleY = 2

fastconvSlowDualDown = FastDualDownSampled(fastconvSlowDual,
                                           downsampleX, downsampleY)


index1 = 1:downsampleX:n
index2 = 1:downsampleY:m

Sindex = spzeros(n,m)
for ii = 0:round(Integer,(n-1)/downsampleX)
    for jj = 0:round(Integer,(m-1)/downsampleY)
        Sindex[1+ii*downsampleX,1+jj*downsampleY] = 1
    end
end

index = findall(!iszero, Sindex[:]) #find(Sindex[:])
S = sparse(I, n*m, n *m) #speye(n*m, n*m)
Sampling = S[index,:];

sigmaSlowDown = zeros(ComplexF64, size(Sampling)[1])

@time info =  gmres!(sigmaSlowDown, fastconvSlowDualDown, Sampling*rhsSlowDual, maxiter = 10 ) 
# println(info[2].residuals[:])



sigmaDownsampled = reshape(sigmaSlowDown, round(Integer,(n-1)/downsampleX) +1,
                          round(Integer,(m-1)/downsampleY) +1)

figure(17);
clf();
imshow(real(sigmaDownsampled)); colorbar();
title("sigma Slow in coarse mesh")


knots = (collect(1:downsampleX:n), collect(1:downsampleY:m))

itp_real = interpolate(knots, real(sigmaDownsampled), Gridded(Linear()))
itp_imag = interpolate(knots, imag(sigmaDownsampled), Gridded(Linear()))

interpsigma = itp_real[collect(1:n),collect(1:m) ] 
              + 1im*itp_imag[collect(1:n),collect(1:m) ]

figure(15);
clf();
imshow(real(interpsigma)); colorbar();
title("interpolated sigma slow")


normSigmaSlow = maximum(abs.(sigmaSlow))

figure(15);
clf();
imshow(real(interpsigma - reshape(sigmaSlow,n,m))/normSigmaSlow)
colorbar()
title("relative error");
