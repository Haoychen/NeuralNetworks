import tkinter
from PIL import Image
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import PIL
from tkinter import filedialog
import os
import math

def fit(w1,h1,w2,h2):
    # fit image of size w1,h1 in window of size w2,h2
    # outuput : new size of image and ratio
    ratio = min(w2/w1,h2/h1)
    return int(w1*ratio), int(h1*ratio), ratio

def getXYinImage(x, y):
    global w_image,h_image,r_image
    # get coordinate of click from event.x, event.y and send back in the coordinate system of image, plus if the click is inside or outside image.
    return x, y, (w_image*(1-paddingThreshold)>x  and x>w_image*paddingThreshold and h_image*(1-paddingThreshold)>y and y>h_image*paddingThreshold)

def position_resize_to_origin(x,y,ratio):
    # transform coordinate to different ratio
    return [x/ratio, y/ratio]

def recenter_points(p1,p2):
    # from p1 and p2 top & down point on bottle, get center of bottle and height of bottle
    return [int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2)],int(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

class App(Frame):
    # Tkinter GUI to select 2 points on bottle
    def __init__(self, master):
        button = Button(master, text="Rotate", command=self.callback2)
        button.pack()
        Frame.__init__(self, master)
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)
        self.image = ImageTk.PhotoImage(original)
        self.display = Canvas(self, bd=0, highlightthickness=1,width=800,height=800)
        self.image_on_canvas = self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")
        self.display.grid(row=0, sticky=W+E+N+S)
        self.pack(fill=BOTH, expand=1)
        self.bind("<Configure>", self.resize)
        self.bind_all("<Button-1>", self.callback)
        w_image,h_image,r_image = 800,800,1
        self.mainloop()

    def rotate_img(self, event):
        print ("haha")
    def resize(self, event):
        global original
        global w_image,h_image,r_image
        size = (event.width, event.height)
        w_image,h_image,r_image = fit(original.size[0],original.size[1],size[0],size[1])
        image = original.resize((w_image, h_image),Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(image)
        self.display.delete("IMG")
        self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")
        # Make reappear older points
        for j in range(0,i):
            x = points[j][0]*r_image
            y = points[j][1]*r_image
            self.display.delete("pt"+str(j))
            self.display.create_oval(x-3, y-3, x+3, y+3, fill="red",tags="pt"+str(j))

    def callback(self, event):
        global i, points
        x,y,inside = getXYinImage(event.x,event.y)
        if inside:
            self.display.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red", tags="pt"+str(i))
            points.append(position_resize_to_origin(x,y,r_image))
            i+=1
            if i>2: #if all three point are selected 
                self.quit()
    def callback2(self):
        global original, root, namefile
        global w_image,h_image,r_image
        print ("Rotated left 90 degree")
        original = original.transpose(Image.ROTATE_90)
        size = (root.winfo_width(), root.winfo_height())
        w_image,h_image,r_image = fit(original.size[0],original.size[1],size[0],size[1])
        image = original.resize((w_image, h_image),Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(image)
        self.display.delete("IMG")
        self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")
# re-order the points matrix to [top, bottom, water_level]
def reorderPoint():
    points.sort(key=lambda row: row[1])
    levelPoint = points[1]
    points[1] = points[2]
    points[2] = levelPoint


# this function resizes the image based on the given reduceSize variable,
# calculates the water level, generate the new naming for the sized image.
# and saves the new image to the resized folder. 
def crop(namefile,percentageExtra,reduceSize,showResult):
    global original, paddingThreshold, i, points, base, root, app
    i=0
    points=[]
    paddingThreshold = 0.05     # threshold for border to recognize pointer in image -> in order to resize the window without selecting a point
    original = Image.open(namefile)
    root = Tk()
    root.title("Bottle selection")
    Label(root,text="Click on the top, water level, and bottom of the bottle").pack()
    root.wm_attributes("-topmost", 1)
    root.focus_force()
    app = App(root)
    with open("./coordinate/"+ namefile.split(".")[0] +"_coord.txt", "w") as myfile:
        coordinate = (namefile + ","  + str(points[0])  + ","  +  str(points[1])  + ","  +  str(points[2]))
        myfile.write(coordinate)
    print ("3 points selected, GUI stoped")
    root.destroy()

    # reorder the three points recorded. 
    # Order: top, bottom, water level
    reorderPoint()
    # Cropping and resizing image
    centerBottle, hBottle = recenter_points(points[0],points[1])
    correctedHeight = hBottle*(1+percentageExtra)/2
    cropped = original.crop((centerBottle[0]-correctedHeight, centerBottle[1]-correctedHeight, centerBottle[0]+correctedHeight, centerBottle[1]+correctedHeight)).resize((reduceSize,reduceSize), PIL.Image.ANTIALIAS)
    
    # Showing result if asked
    if showResult:
        cropped.show()
    # Calculates the water level: 100*(water_height/bottle_height)
    percentage_calculated = int(100*round((points[1][1] - points[2][1])/(points[1][1] - points[0][1]),2))
    # Round the water level value according to the base number. 
    percentage_calculated = int(base * round(float(percentage_calculated)/base))
    photoStageIndex = fields.index("photoStage")
    lastFieldWithExt = namefile[namefile.find(namefile.split('_')[photoStageIndex+1]):].split('.')
    last_field = ".".join(lastFieldWithExt[0:len(lastFieldWithExt)-1])
    new_name = namefile[0:namefile.find(namefile.split('_')[photoStageIndex])-1] +"_"+str(percentage_calculated)+"_"+str(reduceSize)+"x"+str(reduceSize)+"_" + last_field
    
    directory = os.path.dirname(os.path.abspath(__file__))+'/resized'
    if not os.path.exists(directory):           # create new directory if needed
        os.makedirs(directory)
    # Save new image
    print ("selected: " + namefile)
    print ("saved: " + new_name +".png")
    cropped.save(directory+'/'+new_name +".png")

######################################## 
####### Make Changes if Needed #########  
########################################  
# define variables - change values accordingly 
percentageExtra = 0.1      # percentage of extra height to consider for the bottle
reduceSize = 512            # size of reduced image : reduceSize*reduceSize
base = 1                   # this variable defines the water level step size, i.e. base=5, waterLevel = 0, 5, 10, 15, 20, 25... 
fields = ["cuid", "bottleType", "liquidType","percentageFilled", "photoStage", "photoNumber"] 
######################################## 
######################################## 
########################################  

root1 = Tk()
root1.withdraw()
root1.update()
filez = filedialog.askopenfilenames(title='Choose a file')
files = root1.tk.splitlist(filez)
#print(filez)
root1.destroy()
if not os.path.exists("./coordinate/"):
    os.makedirs("./coordinate/")
#naming: cuid_bottleType(_liquidType)_percentageFilled_photoStage_number
#i.e yc3096_cokeBottle_WaterLiquid_25_native_1010
namefiles=[f.split('/')[-1] for f in files]
#extract all files selected by the user. 
for namefile in namefiles:
    if (len(namefile.split('_')) == len(fields)):
        crop(namefile,percentageExtra,reduceSize,showResult=False)
    else:
        print (namefile + ' failed due to inappropriate naming')
        print ('expected fields: '+ str(len(fields)))
        print ('actual fields: ' + str(len(namefile.split('_'))) + "  " +str(namefile.split('_')))
