#### Адаптивный порог кумулятивной энтропии:
#### новый подход к обнаружению DDoS-атак в
#### устройствах Интернета вещей и системах
#### умных домов

Амит Кумар Джайсвал,
*аспирант/исследователь,*
*кафедра радиотехники и кибернетики,*
*Московский физико-технический институт (МФТИ),*
*ул. Институтский, д. 9, Москва, 141701, Российская Федерация,*
`Email:dzhaisval.a@phystech.edu`

**UDC:** 0 04.**Аннотация**
С ростом популярности систем «умного дома» в повседневной жиз-
ни, атаки типа кибер-флудинга на эти взаимосвязанные устройства
стали критически важными. В настоящем исследовании рассматрива-
ется инновационная модель с использованием адаптивного порога, ко-
торая применяет кумулятивный энтропийный анализ временных ря-
дов данных для более эффективного обнаружения и смягчения атак
флудинга в среде «умного дома». Модель устанавливает динамиче-
ские пороги, адаптируемые к изменениям колебаний данных в режиме
реального времени, используя кумулятивную энтропию, показатель,
который определяет непредсказуемость и дисперсию моделей сете-
вого трафика. Будут дополнительно изучены передовые методы ма-
шинного обучения для уточнения процесса установления пороговых
значений, что в конечном итоге приведет к более высокой точности
обнаружения аномалий. Фактически, будут проанализированы такие
важные факторы, как временные паттерны, типы протоколов и дей-
ствия пользователей, с точки зрения их влияния на показатели целей.
Исследование направлено на подтверждение эффективности предла-
гаемых адаптивных пороговых рамок в ответ на значительное сокра-
щение ложных срабатываний при одновременном улучшении реаги-
рования на возникающие угрозы, что в целом повысит устойчивость
систем умного дома к обнаруженным атакам типа «флуд». Преды-
дущая работа будет сосредоточена на адаптации алгоритмов и изу-
чении масштабируемости в различных архитектурах умного дома в
качестве продолжения данной работы. Исследование также нацелено
на решение вопросов, связанных с конфиденциальностью данных и
эффективностью системы.
**Ключевые слова:** Адаптивный порог, кумулятивная энтропия,
анализ временных рядов, смягчение последствий атак типа «флуд»,
безопасность умного дома, обнаружение аномалий, анализ сетевого
трафика, временные паттерны данных

<||WXb23TXrUn3Rxz00yNNr89HV||>##### 1 ##### Introduction

The increasing penetration of smart home systems in everyday life has brought
about the enormous advantages of convenience and automation. However, this
evolutional process also expose the shortcomings by vulnerability issues primarily
focusing on possible cybersecurity threats, one of which is the Distributed Denial
of Service (DDoS) attack that can immensely menace the operation of smart
devices with potential risks to users’ safety and privacy. ThereFore, it is extremely
imperative to develop efficient countermeasures to timely detect and react upon
this type of cyberattack [1,2]. A promising direction towards ameliorating smart
home security is to adopt adaptive thresholding techniques relying on cumulative
entropy based time series analysis. Entropy, a fundamental concept taken from
inFormation theories, measures uncertain or random characteristics in given
dataset. In network traffic analysis domain For instance, monitoring inherent
entropy regimes facilitates distinguishing normal patterns from unusual behaviors
underlying DDoS attacks. By employing cumulative entropy measures, researchers
can develop adaptive methods that adjust thresholds dynamically based on real-
time data, thereby improving detection accuracy and reducing false positives [3].

##### 2  Background on Topic

In recent years, the increasing popularity of smart home systems has raised
serious concerns about cybersecurity, particularly related to distributed denial of
service (DDoS) attacks that can severely overflow network resources, incapacitate
smart devices, and pose threats to user security and privacy [4,5]. ThereFore,
defense mechanisms must be put into places to ensure the resiliency of smart
home systems towards this type of attack. A promising solution can be perceived
by adopting adaptive thresholding techniques based on entropy measures to
analyze time-series data that originate from network traffic [13,14]. Entropy is
a measure that represents the uncertainty or randomness when characterizing
a certain data-set. Specifically, utilizing entropy within the context of network
traffic analysis allows researchers to measure how different are normal network
requests (which are considered as non-malicious) from exploitative counterparts
illustrating DDoS traits (which are deemed malicious)[15][17]. Researchers have
consistently shown how cumulative entropy measures enable detection algorithms
to become more adaptive and accurate due to their capability in adjusting
dynamically with respect to current network conditions [18], [19].

##### 3  Related Works

Several works have studied thresholding techniques For anomaly detection in
network traffic. For example, Sahoo and Arora (2004) proposed a thresholding
technique based on two-dimensional Renyi’s entropy that achieved a much better
segmentation perFormance in image processing applications, indicating the potential
of entropy-based techniques to discriminate normal patterns from anomalies [8].

<||WXb23TXrUn3Rxz00yNNr89HV||>Dragos et al. (2020) investigated some entropy-based metrics For uncertainty
evaluation in Bayesian networks designed For cyber threat detection and concluded
that the entropy measurement is important both in perFormance estimation of
a model and as an added value to decision-making under uncertainty [5].
This work paves the way For applying two-pronged on-line entropy based
defense mechanism at DDoS attack by defending attack traffic in path [7].
Recent improvements in adaptive thresholding techniques show the potential
of such methods in many domains. A machine learning-aided entropy-based
anomaly detection framework for dynamic network adaptations was proposed
by Timcenko and Gajin (2021) [6].
They elevate the need for adaptivity that relies on threshold adjustments by
real-time data analysis, which is essential in combating DDoS attacks in smart
homes [9].
The use of cumulative entropy in time series analysis has been presented
in some previous works. In particular, some researches have focused on using
cumulative residual entropy as a risk measure and they have proven that it is a
useful tool in many different situations. This is consistent with our study’s goal
to apply cumulative entropy for adaptive thresholding in the analysis of time
series data of DDoS attacks [10,11]. Zhang et al. [12] conducted a comprehensive
survey on network anomaly detection frameworks based on kinds of entropy
measures such as Shannon and Renyi entropies and concluded that using many
kinds of features can improve the accuracy of model to against various anomalies
types.

##### 4  Detailed Raw Dataset Description Used in this
##### Research

The UCM_FibIoT2024 dataset gathers substantial data to understand better
Distributed Denial of Service (DDoS) attacks against smart home central control
units, namely the Fibaro Home Center 3. This dataset records many types
of DDoS assaults, such as TCP SYN floods, ICMP floods, and HTTP floods,
to provide light on how they influence the operation and availability of IoT
devices[16]. Data was collected on a local network using the hping3 tool For
SYN and ICMP flood attacks, and the LOIC tool for HTTP flood assaults.
Wireshark software was used to gather network traffic, and the information
is available in PCAP and CSV Formats For future analysis. The data collected
includes critical details such as timestamps, source and destination IP addresses,
protocols, packet lengths, and port numbers [16]. The major purpose of this
dataset is to make it easier to simulate and analyze DDoS attacks on smart
home central control units, hence serving as a resource For cybersecurity and
IoT device protection researchers. Researchers can discover attack patterns,
understand the dynamics of various Forms of DDoS attacks, and design effective
mitigation systems by inspecting network traffic records and packet captures
[16]. The collection is structured to provide comprehensive logs For each attack,

<||WXb23TXrUn3Rxz00yNNr89HV||>such as start and finish timings, frame numbers, and the total number of assault
packets. For simplicity of usage, the data is sorted into folders, and the SYN
flood attack data is further split by the ports targeted (80, 443, and 500)[16].
The UCM_FibIoT2024 dataset serves as a profitable instrument For analyzing
and creating resistance against DDoS attacks on IoT gadgets. It gives a viable
asset For analysts and cybersecurity experts to successfully reenact, analyze,
and moderate DDoS attacks [16]. For more information about the dataset,
refer to the UCM_FibIoT2024 dataset available at [https://doi.org/10.17632/](https://doi.org/10.17632/p42xjtv8pv.1)
[p42xjtv8pv.1](https://doi.org/10.17632/p42xjtv8pv.1)[. However, For this study, we will be only using HTTP flood and](https://doi.org/10.17632/p42xjtv8pv.1)
ICMP flood data.

![a series of photographs showing different types of signs](outputs\template_en_russian (1) (1)_image_4_1.png)

Рис. 1: Flow Chart of the Raw Data Structure Files capture[17]

Таблица 1: Overview of Raw CSV Dataset Columns[16]
**Column Name** **Description**
No. Frame number.
Time Date and time of capture (dd.mm.yyyy hh:mm:ss).
Source Source IP address of the packet.
Destination Destination IP address of the packet.
Protocol Protocol type identifying the network protocol used For each packet.
Length Packet length in bytes.
Source port Source port of the packet.
Destination port Destination port of the packet.

<||WXb23TXrUn3Rxz00yNNr89HV||>##### 5 ##### Research Objective

It should be noted here that the previous authors have utilized several thresholding
techniques in different fields and data sets For research For different purposes.
However, no authors have used our novel approach, that is, network traffic
detection with time series analysis using the cumulative entropy method with
thresholding, to detect such attacks most likely in DDoS on smart home systems
and on IoT devices specifically, which ultimately will help future research scope
growth.

##### 6  Methodologies Used in this research

6 **.1** **Raw Data Preprocessing**

Table 1 represents column names and descriptions of these featured columns
in Raw data preprocessing plays a very crucial and vital role and methods
For this research. There were many steps performed on raw dataset For data
preprocessing. First step was data cleaning, in this step we identified and rectified
errors, inconsistencies, and inaccuracies in the raw dataset. We found source
port and destination port were having huge number of missing, inconsistent and
inaccurate values. We used techniques like handling missing data and removing
duplicates to clean the dataset. Later we analyzed and and removed source
port and destination port due to their high inaccuracy and irrelevancy to this
research outcome.

6 **.2** **Exploratory Data Analysis**

In this section, we will demonstrate the exploratory data analysis performed
by us For more in depth analysis of the dataset. First, we applied many data
analysis codes and functions, we checked the size of the dataset, the description
of the dataset in terms of counts, min and max values, and different percentiles of
each column of the dataset. There are two types of attack files used, one HTTP
flood, a type of attack that targets web servers by overwhelming them with
high HTTP requests. Another file is ICMP Flood, an attack that sends a large
number of ICMPv6 packets (often ping requests, but in this research, data hping
was used for a more aggressive attack) to a target, consuming bandwidth and
resources. In Table 2, we found that, HTTP flood has a higher number of packet
counts but the lowest frequency, whereas ICMPv6 flood has a higher frequency
in comparison to HTTP flood. These findings results in ICMPv6 floods having
a higher effect in the consumption of storage and bandwidth uses, which can
result in a DDoS attack in IoT devices and smart home systems.
In Table 3, we analyzed and found that IP address 10.0.1.22 has the highest
number of traffic as a source IP. HTTP floods have 132 unique values whereas,
ICMPv6 have the highest number of uniqueness in the traffic. As a finding,
higher uniqueness in traffic can have unknown distributed sources which can
result into a DDoS attack(Distributed Denial of service attack) in the network.

<||WXb23TXrUn3Rxz00yNNr89HV||>Таблица 2: Summary of HTTP and ICMPv6 Flood Protocol Features

**Feature** **HTTP Flood** **ICMPv6 Flood**
Count 22,780,665 11,203,Unique 121 Top Protocol TCP ICMPvFrequency 8,060,484 11,172,In table 4, we analyzed and find the destination IP address request traffic and we

Таблица 3: Overview of HTTP and ICMPv6 Flood Source Features

**Feature** **HTTP Flood** **ICMPv6 Flood**
Count 10,799,707 11,203,Unique 132 11,214,Top Source 1 0.0.1.22 1 0.0.1.Frequency 4,859,392 6,found that the count of packet traffic was having the same count. We discovered,
ICMPv6 requests, indicating a slightly higher volume compared to HTTP floods,
with much higher frequency than source IP requests. Finally higher percentage
of requests indicates more aggressive and sustained attack on target.

Таблица 4: Overview of HTTP and ICMPv6 Flood Destination Features

**Feature** **HTTP Flood** **ICMPv6 Flood**
Count 10,799,707 11,203,Unique 127 1,Top Destination 1 0.0.1.22 1 0.0.1.Frequency 5,911,972 11,174,##### 7 ##### Feature Engineering

We did feature engineering task to enrich the features and applied stratified
sampling technique on HTTP flood dataset and ICMPv6 flood dataset, stratified
sampling, which involves dividing the population into subpopulations(strata)
based on one or more common attributes; strata membership is determined by
some factor(s) that are hypothesized to be related to the process being measured,
such as class labels- to reduce business and increase the perFormance of model
learning and testing.
The UCM_FibIoT2024 dataset records are enormous (millions of records),
processing whole CSV file at once requires more time-consuming and computation
resources. ThereFore, we applied stratified random sampling to HTTP flood and
ICMPv6 flood CSV files For our experiment. We have considered the sample
of frac = 0.02 For each file. In algorithm 1, we have demonstrated pseudo code

<||WXb23TXrUn3Rxz00yNNr89HV||>**Algorithm 1** Sampling from Dataset by Protocol

1: **BEGIN**

2:
DATASET *←* Load dataset
3:
GROUPED_DATASET *←* GROUP DATASET BY ’Protocol’
4:
SAMPLED_DATA *←* []
5:
**For** each GROUP **in** GROUPED_DATASET **DO**
6:
SAMPLE *←* SAMPLE 20% FROM GROUP
7:
SAMPLED_DATA.APPEND(SAMPLE)

8:
**END For**
9:
FINAL_SAMPLED_DATA *←* Convert SAMPLED_DATA TO DataFrame
10: **END**

representation For our code and method used on stratified sampling of high
volume dataset. We have also used the time-based feature engineering method
to extract each feature from the time column in separate columns (year, month
day, hour, minutes, seconds, microseconds).

7 **.1** **Cyclical Encoding of Time**

Further we used Cyclical Encoding technique to create more features on time. To
handle the cyclical nature of time (e.g., hours in a day, days in a week), we have
converted time into a circular representation using sine and cosine functions. Let
hour be the hour of the day (in 24-hour Format). The angle can be calculated
as:

angle =
 hour


*×* 2 *π* (1)

The sine and cosine transFormations are defined as follows:

*X* = sin( angle ) = sin
 hour


*×* 2 *π*

(2)

*Y* = cos( angle ) = cos
 hour


*×* 2 *π*

(3)

These transFormations allow the model to capture the cyclical nature of
time, effectively treating 23 : 00 and 00 : 00 as close to each other.

7 **.2** **Seconds Since Epoch**

The term ’seconds since the epoch’ represents the representation of time by
counting the aggregate number of seconds elapsed, from a particular starting
point in time, called the epoch. Seconds since epoch are prominently used For
detection and analysis of Distributed Denial of Service (DDoS) attacks. Here, we
highlight its usage concerning time-stamping and network traffic monitoring. In
DDoS detection systems, every packet of network traffic can be timestamped in

<||WXb23TXrUn3Rxz00yNNr89HV||>seconds since epoch Format to keep the record of exactly when it was received.
Accurate timestamps can be used to track trends, such as a traffic spike over an
extended period of time indicating a possible DDoS attack. Using the timestamps
from incoming packets detection systems can measure the number of packets or
amount of traffic within a given(40 Second) window, if too many packets arrive
within that time span. Systems can compare the number of packets received
in an epoch to a threshold value and then generate an alert if the packet
total is above a pre-determined threshold baseline, indicative of DDoS. So, We
created new feature called seconds since epoch. To do this we have created a
mathematical Formulae calculation For calculating seconds since epoch on each
packet traffic. Let *T* represent the timestamp from the data sample, and let *T* denote the epoch time defined as:

*T* 0 = Timestamp (2024 *,* 1 *,* 1 *,* 0 *,* 0 *,* 0 ) (4)

The Seconds Since Epoch can be calculated as follows:

SecondsSinceEpoch = ( *T* *−* *T* 0 ) *÷* 1 second (5)

Where:

- *T* = [’Time’] (the dataset timestamp features)

- *T* 0 = Timestamp representing the epoch

- The division by 1 second effectively converts the time difference from aTimedelta object into an integer representing seconds.

The Formula presented provides a clear method For calculating Seconds Since
Epoch, which is fundamental in various applications involving time series analysis
and event logging.

##### 8  Mathematical Formulations For calculating entropy
##### and detecting anomalies in packet data

8 **.1** **Entropy Calculation**

The Shannon entropy *H* ( *X* ) For a discrete random variable *X* is defined as:

*H* ( *X* ) = *−*

*n*
X

*i* =*p* *i* log 2 ( *p* *i* )
(6)

Where:

- *p* *i* is the probability of occurrence of the *i* -th outcome.

- *n* is the total number of distinct outcomes.
<||WXb23TXrUn3Rxz00yNNr89HV||>In this context, the entropy is calculated For packet lengths over a rolling
window of size 10:

*H* ( Length window ) = *−*

*m*
X

*j* =*p* *j* log 2 ( *p* *j* )
(7)

Where:

- *m* is the number of distinct packet lengths in the current window.
8 **.2** **Cumulative Entropy Calculation**

The cumulative entropy at time *t* can be expressed as:

*C* ( *t* ) =

*t*
X

*i* =*H* ( Length *i* ) (8)

Where:

- *C* ( *t* ) is the cumulative entropy up to time index *t* .

- *H* ( Length *i* ) is the entropy calculated For packet lengths at time index *i* .
8 **.3** **Anomaly Detection**

Anomaly detection is perFormed using a simple thresholding method. The threshold
*T* is defined as:
*T* = *µ* + 3 *σ* (9)

Where:

- *µ* is the mean of the cumulative entropy values.

- *σ* is the standard deviation of the cumulative entropy values.
An anomaly occurs when:

*A* ( *t* ) =

(
1 if *C* ( *t* ) *> T*
0 otherwise
(10)

Where:

- *A* ( *t* ) indicates whether an anomaly is detected at time index *t* .
<||WXb23TXrUn3Rxz00yNNr89HV||>##### 9 ##### Experiments

we extracted time components from time feature. we have written our own
code For date-time feature with (Year, Month, Day, Hour, Minutes, Seconds,
Microseconds). We used python prebuilt library called DATETIME. In algorithm
2 Label of the algorithm is "Extract Time Components from Date-time which
tells us that this algorithm is responsible to extract specific-time-related features
from date-time. In the beginning of the algorithm there is a comment saying
that: dataset is a data-structure (like table or Data Frame) that has a column
called Time which contains date-time values. A separate loop goes throw all
rows in dataset one by one and extract time column value and stores it into new
created separate column named (Year, Month, Day, Hour, Minutes, Seconds,
Microseconds). The algorithm is then finished with an “END” Statement. In
general, this algorithm is designed to extract all the possible individual time
components including year, month, day, hour, minute, second and microsecond
of a date-time object separately in order to analyze or process them individually.
It can be very helpful For data analysis purposes when we may want to analyze/visualize
some patterns at year/month/day/hour/minute/second/microsecond level or
want to filter/groupBy on based on these individual time components etc. while
perForming some machine learning tasks over timeseries like feature engineering.
We have used mathematical Formulae For sine and cosine Calculations For
Hour, For each row, it retrieves the value of Hour and assigns it to hour value. It
then fetches the value of Hour and stores it in an hour variable. It subsequently
calculates sine and cosine of this Hour value using above mentioned Formulas
1, 2 and 3 as beFore. By doing these calculations, it maps the respective hour
into a Form of cyclic representation which helps to present time-concept to
the models. Sine and Cosine Calculations For Month, Similarly, it fetches the
value of Month and stores it in a month variable. Then, it calculates sine and
cosine For month value with Formulas similar to hours but divided by 12. The
algorithm ends with an “END” Statement representing that all calculations have
been made here. This algorithm essentially perForms conversion of cyclical time
data (hours & months ) into simple sine-cosine way. This whole code is classically
inspired from https://en.wikipedia.org/wiki/Besselpublication by Don E Knuth
which approximates values of sin() & cos().
In algorithm 3, the algorithm name is “Calculate Seconds Since Epoch”. The
algorithm defines the constant EPOCH_TIMESTAMP as a string, representing
this epoch: “2024-01-01 00:00:00”. It loops through each data row assuming that
there is a column with date-times Time in the dataset. For each row, it assigns
the current timestamp from column Time to CURRENT_TIMESTAMP. It
calculates the difference CURRENT_TIMESTAMP minus EPOCH_TIMESTAMP
as TIME_DIFFERENCE. This difference represents how much time passed
between the epoch and this timestamp. The algorithm converts this value then
into seconds by dividing it by one second (which might be implicit for many
programming languages if you handle simply date objects). The resulting number
of seconds since the epoch SECONDS_SINCE_EPOCH, it saves in an additional
column named SecondsSinceEpoch, defined in memory for the dataset data_sample

<||WXb23TXrUn3Rxz00yNNr89HV||>**Algorithm 2** Extract Time Components from Datetime

1: **BEGIN**
2: // Assume data_sample is a data structure (like a table or DataFrame)
with a column ’Time’ of datetime type
3: // Extract year from the ’Time’ column
4: **for** each row in data_sample **do**
5:
row[’Year’] *←* EXTRACT_YEAR(row[’Time’])
6: **end for**
7: // Extract month from the ’Time’ column

8: **for** each row in data_sample **do**
9:
row[’Month’] *←* EXTRACT_MONTH(row[’Time’])
10: **end for**
11: // Extract day from the ’Time’ column
12: **for** each row in data_sample **do**
13:
row[’Day’] *←* EXTRACT_DAY(row[’Time’])

14: **end for**
15: // Extract hour from the ’Time’ column
16: **for** each row in data_sample **do**
17:
row[’Hour’] *←* EXTRACT_HOUR(row[’Time’])
18: **end for**
19: // Extract minute from the ’Time’ column

20: **for** each row in data_sample **do**
21:
row[’Minute’] *←* EXTRACT_MINUTE(row[’Time’])
22: **end for**
23: // Extract second from the ’Time’ column
24: **for** each row in data_sample **do**

25:
row[’Second’] *←* EXTRACT_SECOND(row[’Time’])
26: **end for**
27: // Extract microsecond from the ’Time’ column
28: **for** each row in data_sample **do**
29:
row[’Microsecond’] *←* EXTRACT_MICROSECOND(row[’Time’])
30: **end for**

31: **END**

<||WXb23TXrUn3Rxz00yNNr89HV||>**Algorithm 3** Calculate Seconds Since Epoch

1: **BEGIN**

2:
EPOCH_TIMESTAMP *←* "2024-01-01 00:00:00"
3:
**For** each row in data_sample **DO**
4:
CURRENT_TIMESTAMP *←* data_sample[’Time’][row]
5:
TIME_DIFFERENCE *←* CURRENT_TIMESTAMP

---
EPOCH_TIMESTAMP
6:
SECONDS_SINCE_EPOCH *←* TIME_DIFFERENCE // 1 second

7:
data_sample[’SecondsSinceEpoch’][row] *←*
SECONDS_SINCE_EPOCH
8:
**END For**
9: **END**

at corresponding row. Finally, there is an “END” after which we know that all
these operations end. The main aim of converting date-time values into such a
standardized numeric Format (seconds since epoch) is facilitating their usage
for various operations and especially mathematical analyses during which we
want to help computer somehow understand how timestamps are big/small or
older/newer than other timestamps. For example when comparing them during
some model learning.

9 **.1** **Anomaly Detection Using Threshold and Cumulative**
**Entropy**

In algorithm 4, We experimented on sample data using cumulative entropy and
different threshold values. The Algorithm takes sample data as a Data-frame
having several columns as input, displays the cumulative entropy and which
packets are considered an anomaly. A list of required column names (required
columns) is created, it consists of the attributes, for example, Length, Year,
Month etc to ensure that the dataset contains all the necessary information For
analysis. Then it checks if all provided columns exist in sample data if any of
required column is missing from dataset, then raise Value Error with suitable
message. A function calculate_entropy(data) is created to compute Shannon’s
entropy of given data it calculates normalized value counts of unique values in
the data. It returns the entropy using the Formula mentioned in 6, 7, 8, 9, and earlier in sections of this paper. A new column, PacketLengthEntropy, is created
in sample data. This column stores the rolling entropy calculated over the last
10 entries of the Length column, using the previously defined function. The
cumulative sum of the PacketLengthEntropy column is calculated and stored in
a new column CumulativeEntropy. This serves as the cumulative entropy over
time. Any NaN value in the CumulativeEntropy column is replaced with 0, such
that subsequent calculations do not fail. The threshold to determine anomalies is
computed as mean(CumulativeEntropy) + 3 * standard_deviation(CumulativeEntropy),
where an anomaly represents an entry being seen after which its cumulative
entropy becomes larger than this threshold. Also, another new column Anomaly

<||WXb23TXrUn3Rxz00yNNr89HV||>among the sample dataset constructed by replicate indicating if each packet’s
cumulative entropy exceeds the determined threshold (TRUE for anomaly; FALSE
otherwise). Finally, we print columns year, month, day, hour, minute , second ,
Cumulative_Entropy and Anomaly from our sample datasets.

**Algorithm 4** Anomaly Detection in Packet Data

1: **BEGIN**
2: // Input: *sample* --- *data* (DataFrame containing packet data with required
columns)
3: // Output: Display of cumulative entropy and detected anomalies

4: // Step 1: Define required columns
5: *required* --- *columns* *←* [ *Length, Y ear, Month, Day, Hour, Minute, Second, Microsecond, Protocols...* ]
6: // Step 2: Check if all required columns are present NOT ALL( *col* *∈*
*sample* --- *data.columns* For *col* in *required* --- *columns* )
7: RAISE ValueError("Missing required columns in the dataset.")

8: // Step 3: Define function *calculate* --- *entropy* ( *data* )
9: **Function** *calculate* --- *entropy* ( *data* )
10:
// Calculate value counts of data normalized to probabilities
11:
*value* --- *counts* *←* *COUNT* ( *occurrences* of each unique value in *data* )
12:
RETURN *−* ^P^ ( *p* *i* *∗* log 2 ( *p* *i* + *ϵ* )) where *ϵ* = 1 *e* *−* 13: // Step 4: Create new column *PacketLengthEntropy*

14: *sample* --- *data* [ *^′^* *PacketLengthEntropy* *^′^* ] *←* APPLY *calculate* --- *entropy* ON
ROLLING WINDOW OF SIZE 10 OVER *icmp* --- *sample* --- *data* [ *^′^* *Length* *^′^* ]
WITH min_periods = 15: // Step 5: Calculate cumulative entropy
16: *sample* --- *data* [ *^′^* *CumulativeEntropy* *^′^* ] *←* CUMULATIVE SUM OF
*sample* --- *data* [ *^′^* *PacketLengthEntropy* *^′^* ]

17: // Step 6: Fill NaN values in Cumulative Entropy with zero
18: FILL NaN VALUES IN *sample* --- *data* [ *^′^* *CumulativeEntropy* *^′^* ] WITH 19: // Step 7: Define threshold for anomaly detection
20: *threshold* *←* *MEAN* ( *sample* --- *data* [ *^′^* *CumulativeEntropy* *^′^* ]) + 3 *∗*
*STD* ( *sample* --- *data* [ *^′^* *CumulativeEntropy* *^′^* ])
21: // Step 8: Create new column *Anomaly*

22: *sample* --- *data* [ *^′^* *Anomaly* *^′^* ] *←* TRUE IF *icmp* --- *sample* --- *data* [ *^′^* *CumulativeEntropy* *^′^* ] *>*
*threshold* ELSE FALSE
23: // Step 9: Display results
24: PRINT SELECTED COLUMNS ( *Y ear, Month, Day, Hour, Minute, Second, CumulativeEntropy, Anoma*
25: **END**

<||WXb23TXrUn3Rxz00yNNr89HV||>##### 10 ##### Results and Findings

1 **0.1** **Time series analysis research findings on dataset comparing**
**HTTP flood attacks and ICMP flood attacks**

In this research, we have created a graph for both HTTP flood attack and ICMP
flood attack IOT datasets. we used the most important length and time features
indicated in the dataset. Then we compared these two graphs and we discovered
that ICMP traffic was much higher in the ICMP flood data set. In Fig. 2, we
also found that in the HTTP flood dataset, the other protocol traffic was higher
and stable, which indicates a much lower risk. In Fig. 3, however, in the ICMP
flood dataset, the other protocol traffic was less and unstable, which indicates
a much higher risk. In this time series analysis, we have also found that having
higher ICMP traffic in ICMP floods would have resulted in disrupting other
protocol traffic in the system, creating a traffic congestion in IOT devices and
smart home systems. We also discovered a very important understanding with
this research analysis that, if both ICMP flood attack and HTTP flood attack
have been initiated simultaneously, if both ICMP and HTTP traffic increase
simultaneously, this may suggest a multi-vector attack strategy and would have
and in future can have much more higher risk of traffic congestion and will result
in more successful DDoS attack.

0 200 400 600 Seconds Since Epoch +1.711393ePacket Length

ICMP and ICMPv6 Traffic vs Other Protocols in HTTP flood

Protocol

ARP
BOOTP
CLASSIC-STUN
CoAP
DNS
Gryphon
HTTP
ICMP
ICMPvIGMPvIGMPvIRC
ISAKMP
LLDP
MDNS
NBNS
NTP
QUIC
RRB
SMB
SSDP
SSH
SSL
TCP
TLSv1.UDP
XID
ICMP Traffic
ICMPv6 Traffic

Рис. 2: Time series analysis on HTTP Flooded attack traffic

| | Protocol
ARP
BOOTP
CLASSIC-STUN
CoAP
DNS
Gryphon
HTTP
ICMP
ICMPvIGMPvIGMPvIRC
ISAKMP
LLDP
MDNS
NBNS
NTP
QUIC
RRB
SMB
SSDP
SSH
SSL
TCP
TLSv1.UDP
XID
ICMP Traffic |
|--|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

<||WXb23TXrUn3Rxz00yNNr89HV||>0 200 400 600 Seconds Since Epoch +1.711391ePacket Length

ICMP and ICMPv6 Traffic vs Other Protocols in ICMP flood

Protocol

ARP
BOOTP
CLASSIC-STUN
CoAP
DNS
Gryphon
HTTP
ICMP
ICMPvIGMPvIGMPvLLDP
MDNS
NBNS
NTP
QUIC
RRB
SMB
SSDP
SSH
TCP
TLSv1.UDP
XID
ICMP Traffic
ICMPv6 Traffic

Рис. 3: Time series analysis on ICMP Flooded attack traffic

1 **0.2** **Anomaly Detection using Cumulative Entropy and**
**Thresholding Comparison**

In both graphs, you will find a common chart, "Anomaly Detection Threshold."it
is generally obtained by statistically calculating (taking average and standard
deviation of cumulative entropy values) from the historical data. Now when your
cumulative entropy exceeds that threshold, then there is an indication that an
anomaly has been detected, i.e., some malicious activity (here DDoS attack)
might be going on. Then, in both graphs, you can see a few explicit points
marked where anomalies were detected throughout the time frame for which
the analysis was done. So those peaks in Cumulative Entropy give an idea of
which explicit timings during that period traffic was abnormal with respect to
other timing instances.
In fig 4, we have found that there was an unstable traffic anomaly detected
after the threshold mentioned in the ICMP flooded attack IoT dataset. We
have also discovered, that there was a sudden, unstable traffic change, and an
anomaly was detected after a certain point of threshold calculated. In fig 5,
however, We have investigated the HTTP flooded attack IoT dataset withthe
same threshold algorithm and we found no anomaly detection of any unstable
traffic in comparison to the ICMP flooded attack. We have also discovered that
the traffic did not able to even touch the threshold mark at any point. Hence,
after all the investigation and analysis of both the graphs, we have concluded
that, ICMP flooded attack poses much higher risk in IoT devices and smart
systems than the HTTP flooded attack.

| | Protocol
ARP
BOOTP
CLASSIC-STUN
CoAP
DNS
Gryphon
HTTP
ICMP
ICMPvIGMPvIGMPvLLDP
MDNS
NBNS
NTP
QUIC
RRB
SMB
SSDP
SSH
TCP
TLSv1.UDP
XID
ICMP Traffic |
|--|-----------------------------------------------------------------------------------------------------------------------------------------------------------|

<||WXb23TXrUn3Rxz00yNNr89HV||>
1. 0 0 .5 1 .0 1 .5 2 .0Time Index (Sample Number)
1eCumulative Entropy

Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly Anomaly
ICMP Flooded : Cumulative Entropy Over Time with Anomaly Detection

Cumulative Entropy
Anomaly Detection Threshold
Detected Anomalies

Рис. 4: Anomaly Detection in ICMP Flooded attack dataset

1. 0 0 .5 1 .0 1 .5 2 .0Time Index (Sample Number)
1e1. 1. 1. 1. 1. 1. 1. 1. 1. Cumulative Entropy

1eHTTP Flooded Cumulative Entropy Over Time with Anomaly Detection

Cumulative Entropy
Anomaly Detection Threshold
Detected Anomalies

Рис. 5: Anomaly Detection in HTTP Flooded attack dataset

##### 11  Conclusion

The research findings reported in this article provide substantial information
on the impact of DDoS attacks, with an emphasis on ICMP and HTTP flood
attacks in IoT contexts. We created comparison graphs of network traffic flow
during different assault scenarios by analyzing time series data. Our results

| | Cumulative Entropy | | | | | AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm |
|--|--------------------------------------|-----------|--|--|--|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| | Anomaly Detection
Detected Anomalies | Threshold | | | | AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm |
| | | | | | | |
| | | | | | | AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm |
| | | | | | | AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm |
| | | | | | | AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm |
| | | | | | | AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|--|--|--|--|--|-------------------------------------|-------------|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | Cumulative Entrop
Anomaly Detection | y
Threshold |
| | | | | | | |
| | | | | | Detected Anomalie | s |

<||WXb23TXrUn3Rxz00yNNr89HV||>indicated that the level of ICMP traffic was significantly higher than HTTP in
the ICMP flood data set with the highest risk among other threats as reflected
in our data set. However, HTTP at its constant and highest rate used much
traffic of other protocols within the HTTP flood dataset. ThereFore, the risks
associated with these types of attacks are less in general because they do not
strictly use a specific protocol. By investigating cumulative entropy graphs,
we noted the presence of peak sections or varying windows where anomalies
occur along time signatures which function as reliable indicators to identify a
potential malicious behavior and provide required knowledge on during; how
long and how network can be exposed to DDoS attacks harm. Notably, we
discovered that simultaneous surges in both ICMP and HTTP traffic might
indicate a multi-vector assault approach. This scenario increases the likelihood
of network congestion and may lead to more effective DDoS attacks on IoT
devices and smart home systems. In conclusion, our findings clearly show that
ICMP flood attacks offer a far higher danger to IoT devices and smart home
systems than HTTP flood assaults. The findings of this study will help to design
effective countermeasures For strengthening the security of smart home systems
against growing DDoS attacks. Further research is required to develop anomaly
detection tools and study adaptive responses For increasing resilience to these
sorts of intrusions.

##### 12  Future Work and Ideas

In the future, we will use the same datasets, UCM_fiblo For hybrid model
training on IoT devices and smart home systems datasets. This can be made
possible by exploiting more powerful machine learning methods like deep learning
models to improve anomaly detection accuracy in different kinds of network
domains. Also, the combination with real-time big data analytics and edge
computing aids in immediate threat intelligence-driven response and minimizes
the latency. It is worth collaborating with IoT device manufacturers where
adaptive security functionalities should be directly deployed in devices so that
smart homes will have the capabilities of proactive defense against emerging
DDoS attack vectors. In addition, multiple field experiments on real-world smart
home deployment should be more intense, as these deployments provide a wealth
of practical data that can facilitate rapid iteration on performance optimizations
based on both user feedback and scientific measurements.

##### Acknowledgements

I am grateful to my wife, Darina Olegovna Ershova, For her professional advice
and calm hand help during these challenging times. My study was helped by her
expertise and vision, as well as her meticulous attention to detail. I am grateful
to my department For their valuable contributions to this research, including
their amazing experience of guidance. We appreciate the valuable feedback and

<||WXb23TXrUn3Rxz00yNNr89HV||>recommendations from the faculty of Artificial Intelligence and Cybersecurity
at Moscow Institute of Physics and Technology (MIPT), Department of Radio
Engineering and Cybernetics, (Institutsky Lane, 9, Moscow, Russian Federation,
1 41701). Their contributions helped shape an intellectually diverse conversation
about my work.

##### Список литературы

[1] Lee S-H, Shiue Y-L, Cheng C-H, Li Y-H, Huang Y-F. Detection and
Prevention of DDoS Attacks on the IoT. Applied Sciences. 2022; 12 (23):
1 2407. doi: [https://doi.org/10.3390/app122312407](https://doi.org/10.3390/app122312407 )

[2] Shrahili, M.; Kayid, M. Cumulative Entropy of Past Lifetime for Coherent
Systems at the System Level. Axioms 2023, 12, 899. doi: [https://doi.org/](https://doi.org/10.3390/axioms12090899)
[1 0.3390/axioms12090899](https://doi.org/10.3390/axioms12090899)

[3] M Tharun Kumar, G.Sesha Phaneendra babu, D. Lakshmi Narayana
Reddy, "A Novel Framework for Mitigating DDoS Attacks in IoT
Based Smart Network Environments using Machine Learning," *Industrial*
*Engineering Journal* ISSN: 0970-2555 Volume : 53, Issue 5, May : 2024,
[http://www.journal-iiie-india.com/1_may_24/125_online_may.pdf](http://www.journal-iiie-india.com/1_may_24/125_online_may.pdf)[.](http://www.journal-iiie-india.com/1_may_24/125_online_may.pdf)

[4] A. K. Jaiswal, "Deep Comparison Analysis: Statistical Methods and Deep
Learning For Network Anomaly Detection,"2024. doi: [https://doi.org/10.](https://doi.org/10.5281/zenodo.14051107)
[5281/zenodo.14051107](https://doi.org/10.5281/zenodo.14051107)[.](https://doi.org/10.5281/zenodo.14051107)

[5] J. Dragos, J. P. Ziegler, A. de Villiers, A.-L. Jousselme, and E. Blasch,
"Entropy-Based Metrics For URREF Criteria to Assess Uncertainty
in Bayesian Networks For Cyber Threat Detection,"in *2019* *22nd*
*International Conference on InFormation Fusion (FUSION)* , Ottawa, ON,
Canada, 2019, pp. 1-8, doi: 1 0.23919/FUSION43075.2019.9011276 .

[6] V. Timcenko and S. Gajin, "Machine Learning Enhanced Entropy-
Based Network Anomaly Detection," *Advances in Electrical and Computer*
*Engineering* , vol. 21, no. 4, pp. 51-60, 2021, doi: 1 0.4316/AECE.2021.04006 .

[7] P. Verma, S. Tapaswi, and W.W Godfrey, "An Adaptive Threshold-Based
Attribute Selection to Classify Requests Under DDoS Attack in Cloud-
Based Systems," *Arab Journal of Science and Engineering* , vol. 45, pp.
2813–2834, 2020, doi: 1 0.1007/s13369-019-04178-x .

[8] P. Sahoo and Gurdial Arora, "A Thresholding Method Based on Two-
Dimensional Renyi’s Entropy," *Pattern Recognition* , vol. 37, no. 6, pp. 1149-
1161, 2004, doi: 1 0.1016/j.patcog.2003.10.008 .

[9] H. Lin and N.Bergmann, "IoT Privacy and Security Challenges For Smart
Home Environments," *InFormation* , vol. 7, no. 44, pp., 2016, doi: 1 0.3390/
info7030044 .

<||WXb23TXrUn3Rxz00yNNr89HV||>[10] M.C Dani et al., "Adaptive Threshold For Anomaly Detection Using Time
Series Segmentation,"in *Neural InFormation Processing* , S.Arik et al., Eds.,
vol 9491 of Lecture Notes in Computer Science., Springer Cham., pp., 2015.

[11] Amit Jaiswal., "DOS Attack Network Traffic Monitoring in Software
Defined Networking Using Mininet and RYU Controller,"DOI: 1 0.21203/
rs.3.rs-2282189/v1 , 2022.

[12] Berezi´nski P, Jasiul B, Szpyrka M. An Entropy-Based Network Anomaly
Detection Method. Entropy. 2015; 17(4):2367-2408. DOI: [https://doi.org/](https://doi.org/10.3390/e17042367)
[1 0.3390/e17042367](https://doi.org/10.3390/e17042367)

[13] Rong Lan and Lekang Zhang. 2023. Image Thresholding Segmentation
Algorithm Based on Two-parameter Cumulative Residual Masi Entropy.
In Proceedings of the 2022 5th International Conference on Artificial
Intelligence and Pattern Recognition (AIPR ’22). Association For
Computing Machinery, New York, NY, USA, 406–411, DOI: [https://doi.](https://doi.org/10.1145/3573942.3574041)
[org/10.1145/3573942.3574041](https://doi.org/10.1145/3573942.3574041)

[14] J.Assfalg et al., "Time Series Analysis Using the Concept of Adaptable
Threshold Similarity,"in *18th International Conference on Scientific and*
*Statistical Database Management (SSDBM’06)* , Vienna, Austria , pp.,251-
[260 ,2006 ,doi:](https://www.dbs.ifi.lmu.de/Publikationen/Papers/ssdbm06.threshold.pdf )[https://www.dbs.ifi.lmu.de/Publikationen/Papers/ssdbm06.](https://www.dbs.ifi.lmu.de/Publikationen/Papers/ssdbm06.threshold.pdf )
[threshold.pdf](https://www.dbs.ifi.lmu.de/Publikationen/Papers/ssdbm06.threshold.pdf )

[15] D.Shang and P.Shang , "Analysis of Time Series in the Cumulative Residual
Entropy Plane Based on Oscillation Roughness Exponent,"Nonlinear
Dynamics , vol.,100 ,pp.,2167–2186 ,2020 ,doi: 1 0.1007/s11071-020-05646-y

[16] A.Patharkar et al., "Eigen-entropy Based Time Series Signatures to
Support Multivariate Time Series Classification,"Scientific Reports , vol.,no.,1 Article16076 ,2024 ,doi: 1 0.1038/s41598-024-66953-[17] Huraj, Ladislav; Lietava, Jakub; ˇSimon, Marek (2024),
“UCM_FibIoT2024”, Mendeley Data, V1, doi: 1 0.17632/p42xjtv8pv.[18] Yu, H., Yang, W., Cui, B. et al. Renyi entropy-driven network traffic
anomaly detection with dynamic threshold. Cybersecurity 7, 64 (2024).
[https://doi.org/10.1186/s42400-024-00249-1](https://doi.org/10.1186/s42400-024-00249-1)

[19] M. Thakur and R.K. Sharma, "Anomaly Detection in Smart Home
Networks Using Adaptive Thresholding Techniques Based on Cumulative
Entropy," *International Journal of Computer Applications* , 2022. doi: [https:](https://doi.org/10.5120/ijca2016911955)
[//doi.org/10.5120/ijca2016911955](https://doi.org/10.5120/ijca2016911955)[.](https://doi.org/10.5120/ijca2016911955)

[20] D.G. Narayan, W. Heena, and K. Amit, "A Collaborative Approach
to Detecting DDoS Attacks in SDN Using Entropy and Deep
Learning," *Journal of Telecommunications and InFormation Technology* ,
vol. 3, no. 3, 2024. doi: [https://doi.org/10.26636/jtit.2024.3.1609](https://doi.org/10.26636/jtit.2024.3.1609)[.](https://doi.org/10.26636/jtit.2024.3.1609)

<||WXb23TXrUn3Rxz00yNNr89HV||>