read -p "Are you in a python3 env of choice w/ gdal installed? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pip install pandas
    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install sklearn
    pip install pydot
    pip install seaborn
    pip install opencv-python
    
    # I added Jaspreet's packages 
    pip install pomegranate
    pip install jupyter
    
    pip install ipython
fi

