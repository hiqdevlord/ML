import sys
import pylab
from numpy import exp
from PIL import Image
"""We are goiing to use a MLP with one layer of hidden units to store two classes of images, 
one which contains a + and the other which contains a -. PIL will be used 
to get the pixel values and pass it as input to the MLP. The images considered 
will be 30x30, so there will be <inputs> input units(which should be set to the total
number of pixels in the input images), there will be <hidden> hidden units,
and one output unit, if the image is a + then the output should be 1, otherwise 
it should be 0. The learning factor is set to 0.5"""

img_ht = 50
img_wdt = 50

inputs = img_ht*img_wdt
hidden = 1000

# input layer to hidden layer weights and hidden layer to output layer weights
itoh_wts=[]
htoo_wts=[]

#learning factor
n=0.2


#lets initialize the weights to some really small values like 0.01
for vali in range(hidden):
    temp_list=[]
    for valj in range(inputs):
        temp_list.append(0.05)
    itoh_wts.append(temp_list)

for vali in range(hidden):
    htoo_wts.append(0.05)


#we will use just the sum function at the hidden layer
#at the output layer we will use the sigmoid as the threshold function

#lets get the image first, the testing images should be passed as arguments.
for p in range(1,len(sys.argv)):
    #reinitialize input units and hidden units
    input_units=[]
    hidden_units=[]
    for vali in range(hidden):
        hidden_units.append(0)
    for vali in range(inputs):
        input_units.append(0)
    pre, ext = sys.argv[p].split('.')
    if pre.endswith("plus"):
        result=1
    elif pre.endswith("minus"):
        result=0
    else:
        print "The naming of the test image file is bad, please check"
        sys.exit(0)

    im = Image.open(sys.argv[p])
    pixel_li = list(im.getdata())
    #the getdata return a three valued tuple for each pixel, for our 
    #images all the three values are the same, so taking just one from 
    #them and assigning them to the input units
    for vali in range(inputs):
        input_units[vali] = float(pixel_li[vali][0])
    
    #calculate activations for the hidden units
    for vali in range(hidden):
        for valj in range(inputs):
            hidden_units[vali]=float( hidden_units[vali] + input_units[valj]*itoh_wts[vali][valj])

    #we should remember to print the weights itoh_wts as an image "
    #we are not doing it right now since the PIL functions are 
    #mostly showing overflow errors with long , instead we will plot a graph
    #towards the end of the processing
    
    #calculating the final result
    output = 0
    for vali in range(hidden):
        output = output + hidden_units[vali]*htoo_wts[vali]
    #calculating the sigmoid 
    class_val = 1/(1+exp(-output))
    print class_val
    #if class_val >= 0.5 :
    #   class_val = 1

    if result != class_val:
        if class_val == 1:
            print "Image recognized as plus but is minus"
        else:
            print "Image recognized as minus but is plus"
        print "Updating Weights"    
        #perform back propagation to update the weights
        #the weights have to be updated using the following rules 
        #   change in htoo_wts[i][j] = n*sum_over_i((result[i]-class_val[i])hidden_units[j]
        #   change in itoh_wts[j][q] = n*sum_over(sum_over_ij((result[i]-class_val[i])htoo_wts[i][j])*hidden_units[j]*(1-hidden_units[j])*input_units[q]
        #but in our case there is only one output unit, hence i=0 by default, hence class_val is same as class_val[0] etc, so on
        diff = result-class_val
        # also the equation for updating the first layer weights uses the old value of htoo[][] and not the updated ones, so we 
        # should first update the first layer weigts and then the second layer weights
        #updating first layer weights
        htoo_wts_sum = 0
        for j in range(hidden):
            htoo_wts_sum = htoo_wts_sum + diff*htoo_wts[j]
        for j in range(hidden):
            for q in range(inputs):
                itoh_wts[j][q] = itoh_wts[j][q] + n*(htoo_wts_sum*hidden_units[j]*(1-hidden_units[j])*input_units[q])
        #update second layer weights
        for j in range(hidden):
            htoo_wts[j] = htoo_wts[j] + n*diff*hidden_units[j]

    else :
        if class_val == 1 :
            print "Image correctly recognised as plus"
        else:
            print "Image correctly recognised as minus"

#lets print a graph of the final weights 
y_list=[]
for value in itoh_wts:
    y_list.extend(value)
x_list = range(221,1088)
pylab.xlabel("Weight value")
pylab.ylabel("Actual weights")
pylab.plot(x_list, y_list[221:1088])
pylab.savefig("weight_graph.png")
pylab.show()
