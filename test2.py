import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
import scipy
import math

numSimulations=1000
given_p=10 # number frames per simulation
given_mu = 1
given_lambda = 2

def calculateArrivalTime(currFrameNum, arrivaltimeintervals):
    sum=0
    for i in range(0, currFrameNum):
        sum += arrivaltimeintervals[i]
    return sum
def calculateQueueingTime(currFrameNum, frameData, currFrameArr):
    #get previous frame's end time
    x=frameData[currFrameNum-1][3]
    #get current frame's arrival time
    y=currFrameArr[0]
    diff = x-y
    if diff < 0:
        diff = 0
    return diff
def calculateDepartureTime(currFrameArr):
    sum = 0
    for i in range(0,3):
        sum += currFrameArr[i]
    return sum
def calculateTotalTime(currFrameArr):
    queuingtime = currFrameArr[1]
    servicetime = currFrameArr[2]
    sum = queuingtime + servicetime
    return sum
def runSimulation():
    arrivaltimeintervals = []
    servicetimes = []
    for i in range(0,9):
        arrivaltimeintervals.append(np.random.exponential(0.5))
    for i in range(0,10):
        servicetimes.append(np.random.exponential(1))

    #format of thing: [arrival time, queing time, service time, departure time, total service time (Tsubi)]
    frameData = []
    for i in range(0,len(servicetimes)):
        tmp = [0, 0, 0, 0, 0]
        if i!=0:
            tmp[0] = calculateArrivalTime(i, arrivaltimeintervals)
        if i!=0:
            tmp[1] = calculateQueueingTime(i, frameData, tmp)
        tmp[2] = servicetimes[i]
        tmp[3] = calculateDepartureTime(tmp)
        tmp[4] = calculateTotalTime(tmp)
        frameData.append(tmp)
    # for i in frameData:
    #     print(i)

    return frameData
############################ Part 1-a #################################
def calcW(frameData):
    # total up T sub i for all frames serviced
    W=0
    for i in range (0, len(frameData)):
        W += frameData[i][4]
    return W
def calc_Wbar(ws):
    return np.mean(ws)
############################ Part 1-b #################################
def calcY(frameData):
    sum=0
    for frame in frameData:
        sum += frame[2]
    return sum
def calc_Ybar(ys):
    return np.mean(ys)

def calc_cstar(W_arr, Y_arr):
    if len(W_arr)==1:
        return float("nan")
    cov_matrix_WY =  np.cov([W_arr, Y_arr])
    var_Y = cov_matrix_WY[1][1]
    cov_WY = cov_matrix_WY[1][0]
    return (-cov_WY / var_Y)

def calc_Zbar(W_arr, Y_arr, Y_bar):
    c_star = calc_cstar(W_arr, Y_arr)
    ExpectY = given_p / given_mu
    Z_bar = W_bar + c_star*(Y_bar - ExpectY)
    return Z_bar
############################ Part 1-c #################################
def calcQ(frameData):
    sumS = 0
    for i in range(0,10):
        sumS += frameData[i][2]
    sumI=frameData[9][0]
    return (sumS-sumI)

def calc_Qbar(Q_arr):
    return np.mean(Q_arr)

def calc_Hbar(W_arr, W_bar, Q_arr, Q_bar):
    c_star = calc_cstar(W_arr, Q_arr)
    ExpectQ = (given_p/given_mu) - ((given_p - 1)/given_lambda)

    H_bar = W_bar + c_star*(Q_bar-ExpectQ)
    return H_bar
############################ Part 1-d #################################
def calcL(frameData):#format of thing: [arrival time, queing time, service time, departure time, total service time (Tsubi)]
    L=0
    for k in range(0,10): #for each frame calc frames in sys at arrival time
        # print("On the ", k, " frame:")
        Nki=0
        for itr in range(0,k): #check if prev frames in system at frame i's arrival time
            # print("checking prev frame ", itr)
            if frameData[itr][0]<= frameData[k][0] and frameData[k][0] <= frameData[itr][3]:
                Nki+=1
                # print("it counts...")
        L += ((Nki+1)*(1/given_mu))
        # print("Total frames also in system with frame ", k," : ", Nki)
    # print(L)
    return L
def calc_Lbar(L_arr):
    return np.mean(L_arr)
############################ utilities #################################
def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.nanmean(a),
    se = np.nanstd(a)/math.sqrt(n)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
def calcBForEachRunN(data):
    b_arr=[]
    for i in range (0, len(data)):
        tmp=data[:i+1]
        conf_int =mean_confidence_interval(tmp)
        b=abs(conf_int[1]-conf_int[2])
        b_arr.append(b)
    return b_arr

def getTenPercent(theta):
    tenPercent = 0.1*theta
    # return theta + tenPercent , theta - tenPercent
    return theta+0.1, theta-0.1
def calcN(avg_arr):
    runs=0
    yeet = 0
    m, l, u = mean_confidence_interval(avg_arr)
    for i in avg_arr:
        if l<= i and i<= u and yeet == 1:
            break
        elif l<= i and i<= u:
            runs +=1
            yeet +=1
        else:
            runs+=1
            yeet=0
    return runs
def calcNtwo(avg_arr):
    runs=10
    yeet = 0
    m, l, u = mean_confidence_interval(avg_arr)
    for i in range(10, len(avg_arr)):
        numinrange = 0
        for j in range (i-10, i):
            if l<= avg_arr[j] and avg_arr[j]<= u and numinrange == 2 :
                yeet=1
                break
            elif l<= avg_arr[j] and avg_arr[j]<= u:
                numinrange += 1
            else:
                numinrange +=0
        if yeet == 1:
            break
        runs += 1
    return runs
##Main##
#things I need for each run

W_arr = [] #W for each simulation run
W_bar_arr = [] # running average of W

Y_arr = [] # Y for each simulation run
Y_bar_arr = [] # running avg Y
Z_bar_arr = [] # Zbar for each runSimulation

Q_arr = []
Q_bar_arr = []
H_bar_arr = []

L_arr=[]
L_bar_arr = []

xaxis=list(range(0,numSimulations))

# start the thing
for i in range(0, numSimulations):
    frameData=runSimulation()
    # a
    W=calcW(frameData)
    W_arr.append(W)
    W_bar=calc_Wbar(W_arr)
    W_bar_arr.append(W_bar)

    # b
    Y=calcY(frameData)
    Y_arr.append(Y)
    Y_bar=calc_Ybar(Y_arr)
    Y_bar_arr.append(Y_bar)

    Z_bar = calc_Zbar(W_arr, Y_arr, Y_bar)
    Z_bar_arr.append(Z_bar)

    # c
    Q = calcQ(frameData)
    Q_arr.append(Q)
    Q_bar = calc_Qbar(Q_arr)
    Q_bar_arr.append(Q_bar)
    H_bar = calc_Hbar(W_arr, W_bar, Q_arr, Q_bar)
    H_bar_arr.append(H_bar)
    #d
    L = calcL(frameData)
    L_arr. append(L)
    L_bar = calc_Lbar(L_arr)
    L_bar_arr.append(L_bar)


#
# # plot theestimated value of θ under each of the four estimators as a
# # function of the number of simulation runs,n
# plt.plot(xaxis, W_bar_arr, color = 'tab:blue')
# plt.title("1.a Raw Estimator")
# plt.xlabel("Number of simulations run")
# plt.ylabel("Theta = W bar")
# plt.savefig("1a.png")
# plt.close()
#
# plt.plot(xaxis, Z_bar_arr, color = 'tab:orange')
# plt.title("1.b Control Variate: Service Time")
# plt.xlabel("Number of simulations run")
# plt.ylabel("Theta = Z bar")
# plt.savefig("1b.png")
# plt.close()
#
# plt.plot(xaxis, H_bar_arr, color = 'tab:red')
# plt.title("1.c Control Variate: Difference of Service Time and Interarrival Time")
# plt.xlabel("Number of simulations run")
# plt.ylabel("Theta = H bar")
# plt.savefig("1c.png")
# plt.close()
#
# plt.plot(xaxis, L_bar_arr, color = 'tab:green')
# plt.title("1.d Number of Frames in the System")
# plt.xlabel("Number of simulations run")
# plt.ylabel("Theta = L bar")
# plt.savefig("1d.png")
# plt.close()
#
fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xaxis, W_bar_arr, 'k-', label = "W bar")
ax.plot(xaxis, Z_bar_arr, 'k:', label = "Z bar")
ax.plot(xaxis, H_bar_arr, 'k-.', label = "H bar")
ax.plot(xaxis, L_bar_arr, 'k--', label = "L bar")
plt.title("All estimators")
plt.xlabel("Number of Simulation Runs: " + str(numSimulations))
plt.ylabel("Estimated Value of Theta")
# red_patch = matplotlib.patches.Patch(color='tab:red', label="Estimator: H bar")
# green_patch = matplotlib.patches.Patch(color='tab:green', label="Estimator: L bar")
# orange_patch = matplotlib.patches.Patch(color='tab:orange', label="Estimator: Z bar")
# blue_patch = matplotlib.patches.Patch(color='tab:blue', label="Estimator: W bar")
# plt.legend(handles=[blue_patch, orange_patch, red_patch, green_patch])
legend = ax.legend(fontsize='x-large')
plt.savefig("all.png")
plt.close()
#
#
# # plot the normalized (with respect to the estimated value) 90% confidence
# # interval width, b, under each of the four estimators as a function of the
# # number of simulation runs, n.
#
conf_int_W = calcBForEachRunN(W_bar_arr)

conf_int_Z = calcBForEachRunN(Z_bar_arr)

conf_int_H = calcBForEachRunN(H_bar_arr)

conf_int_L = calcBForEachRunN(L_bar_arr)

#
fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xaxis, conf_int_W, 'k-', label = "W bar")
ax.plot(xaxis, conf_int_Z, 'k:', label = "Z bar")
ax.plot(xaxis, conf_int_H, 'k-.', label = "H bar")
ax.plot(xaxis, conf_int_L, 'k--', label = "L bar")
plt.title("Confidence Intervals for All Estimators")
plt.xlabel("Number of Simulation Runs: " + str(numSimulations))
plt.ylabel("Width of Confidence Interval, b")
legend = ax.legend(fontsize='x-large')
# red_patch = matplotlib.patches.Patch(color='tab:red', label="H bar")
# green_patch = matplotlib.patches.Patch(color='tab:green', label="L bar")
# orange_patch = matplotlib.patches.Patch(color='tab:orange', label="Z bar")
# blue_patch = matplotlib.patches.Patch(color='tab:blue', label="W bar")
# plt.legend(handles=[blue_patch, orange_patch, red_patch, green_patch])
plt.savefig("allCI.png")
plt.close()


print("output final thetas: ")

final_est_W = W_bar_arr[numSimulations-1]
final_est_Z = Z_bar_arr[numSimulations-1]
final_est_H = H_bar_arr[numSimulations-1]
final_est_L = L_bar_arr[numSimulations-1]
print("Theta = ", final_est_W, " for estimator W bar")
print("Theta = ", final_est_Z, " for estimator Z bar")
print("Theta = ", final_est_H, " for estimator H bar")
print("Theta = ",final_est_L, " for estimator L bar")
print()

print("Confidence interval for W bar", mean_confidence_interval(W_bar_arr))
print("Confidence interval for Z bar",mean_confidence_interval(Z_bar_arr))
print("Confidence interval for H bar",mean_confidence_interval(H_bar_arr))
print("Confidence interval for L bar",mean_confidence_interval(L_bar_arr))

#now determine where n is within +- 10% of these vals and plot till that point

runsW = calcN(W_bar_arr)
runsZ = calcN(Z_bar_arr)
runsH = calcN(H_bar_arr)
runsL = calcN(L_bar_arr)

print(runsW, runsZ, runsH, runsL)
f = open("output.txt", "w")
f.write(str(runsW) + "\t" + str(runsZ) + "\t" + str(runsH) + "\t" + str(runsL) + "\n")
f.close()

new_W_bar_arr = W_bar_arr[0:runsW]
new_Z_bar_arr = Z_bar_arr[0:runsZ]
new_H_bar_arr = H_bar_arr[0:runsH]
new_L_bar_arr = L_bar_arr[0:runsL]

print(len(new_W_bar_arr), len(new_Z_bar_arr), len(new_H_bar_arr), len(new_L_bar_arr))

print("Theta after n runs: ", new_W_bar_arr[-1])
print("Theta after n runs: ", new_Z_bar_arr[-1])
print("Theta after n runs: ", new_H_bar_arr[-1])
print("Theta after n runs: ", new_L_bar_arr[-1])

new_conf_int_W = calcBForEachRunN(new_W_bar_arr)
new_conf_int_Z = calcBForEachRunN(new_Z_bar_arr)
new_conf_int_H = calcBForEachRunN(new_H_bar_arr)
new_conf_int_L = calcBForEachRunN(new_L_bar_arr)

fig = plt.figure(figsize=(10, 5))
plt.tight_layout()
ax = fig.add_subplot(1, 1, 1)
ax.plot(list(range(0,runsW)), new_conf_int_W, 'k-',  label="W bar, n = "+ str(runsW))
ax.plot(list(range(0,runsZ)), new_conf_int_Z, 'k:', label="Z bar, n = "+ str(runsZ))
ax.plot(list(range(0,runsH)), new_conf_int_H, 'k-.',label="H bar, n = " + str(runsH))
ax.plot(list(range(0,runsL)), new_conf_int_L, 'k--', label="L bar, n =" + str(runsL))
plt.title("Confidence Intervals for All Estimators")
plt.xlabel("Number of Simulation Runs")
plt.ylabel("Width of Confidence Interval b")
# red_patch = matplotlib.patches.Patch(color='tab:red', label="H bar, n = " + str(runsH))
# green_patch = matplotlib.patches.Patch(color='tab:green', label="L bar, n =" + str(runsL))
# orange_patch = matplotlib.patches.Patch(color='tab:orange', label="Z bar, n = "+ str(runsZ))
# blue_patch = matplotlib.patches.Patch(color='tab:blue', label="W bar, n = "+ str(runsW))
# plt.legend(handles=[blue_patch, orange_patch, red_patch, green_patch])
legend = ax.legend(fontsize='x-large')
plt.savefig("2allCI.png")
plt.close()
