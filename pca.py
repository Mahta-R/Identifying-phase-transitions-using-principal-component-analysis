import numpy as np
import matplotlib.pyplot as plt

number=input("# of configurations=")
file_name_configs_ising="spinConfigs_Ising_L"+number+".txt"
file_name_labels_ising="temperatures_Ising_L"+number+".txt"

file_name_configs_gauge="spinConfigs_gaugeTheory_L"+number+".txt"
file_name_labels_gauge="labels_gaugeTheory_L"+number+".txt"

model=input("Model: (i) or (g)=")

if(model=="i"):
    Spins=np.loadtxt(file_name_configs_ising)
    T=np.loadtxt("temperatures_Ising_L"+number+".txt")
else:
    Spins=np.loadtxt(file_name_configs_gauge)
    T=np.loadtxt("labels_gaugeTheory_L"+number+".txt")

print(Spins.shape)

N_data=Spins.shape[0]
N_spins=Spins.shape[1]
print("There are %d configurations and %d spins in each configuarion" %(N_data,N_spins))

### Shifting data to make the average value 0
for i in range(N_spins):
    average=np.mean(Spins[:,i])
    Spins[:,i]-=average
    if(np.mean(Spins[:,i])>1E-3):
        print("Not zero!")
magnetization=np.zeros(N_data)
for i in range(N_data):
    magnetization[i]=np.mean(Spins[i,:])

X_c=np.array(Spins)

eigen_vals,eigen_vects=np.linalg.eig(X_c.T@X_c/(N_data-1))
eigen_vals*=1./sum(eigen_vals)

X_prime=X_c@eigen_vects

#Visualize
print(X_prime.shape)
index_0=np.where(T==0)
intdex_1=np.where(T==1)
#
plt.scatter(X_prime[:,0],X_prime[:,1],c=T)
plt.show()
if(model=="ising"):
    plt.semilogy(eigen_vals[:10],"o-")
    plt.xlabel("Eigen values")
else:
    plt.plot(eigen_vals[:10],"o-")
    plt.xlabel("Eigen values")
plt.show()
plt.plot(eigen_vects[:,0])
plt.xlabel("Eigen vectors")

plt.show()
plt.scatter(T,magnetization,s=20)
plt.show()

total_explained=np.zeros(len(eigen_vals))
for i in range(len(eigen_vals)):
    total_explained[i]=sum(eigen_vals[:i])
plt.plot(total_explained,"o-")
plt.title("Explained variance")
plt.show()

