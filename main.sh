read -p "Are all packages installed? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Just type y or n. No need to hit enter. "
    echo "First we will run through early ACLED, IPC label, and feature correlation  visualization work."
	read -p "Would you like to see this visualization? " -n 1 -r
    echo
	if [[ $REPLY =~ ^[Yy]$ ]]
    then 
        echo "Great! Getting started..."
		python acled_visualization/ACLED-data-vis.py
	fi
    echo "We can also walk through some Modis image input visualization."
    read -p "Would you like to see this visualization? " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then 
        echo "Great! Getting started..."
        python modis_data/Histogram_Vis.py
    fi
    echo "We are now moving onto the model walk through."
    read -p "If you would like to walk through Amber's models hit y, to go directly to Jaspreet's hit n. " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "Please be aware that there is a long string of warnings I could not repress during this walk through."
        echo "The code is not (/that/) broken :)"
	    echo "We can do a quick start, or see a demo of some preliminary data processing."
	    echo "Answer y to see demo or n to skip."
        read -p "Would you like to see the data processing demo? " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            echo "Great! Getting started..."
            python modis_data/Model_Walk_Through.py -d
        else
            echo "That's fine! Let's get started..."
            python modis_data/Model_Walk_Through.py
        fi
    fi

    echo "Ooowweee. We have now finished Amber's component. We will now start Jaspreet's."
    read -p "Are you ready to continue? " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then 
        echo "Great! Getting started..."
        python acled_visualization/HMM_FAM/Ensemble.py
        python acled_visualization/Anommultivariatehmm.py
        python acled_visualization/Maxmultivariate.py
        python acled_visualization/Transition\ multivariate.py
    else
        echo "You skipped Jaspreet's portion!"
    fi
  	
    echo "Please run amber_setup.sh to install all packages."

fi
echo "You have completed the Famine Intensity guided tour."
echo "Please enjoy this after credits scene."
python boss_level.py
echo "This is entirely Amber's fault."


