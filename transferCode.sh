#!/bin/bash

selectUplaodTarget() {
    echo "Select target computer to uoload:"
    echo "(ka)enguru, (ko)ala, (s)patz, (*)all"
    read TARGET 

    echo "The target computer is: $TARGET"

    case $TARGET in
        "ka" | "kaenguru")
            scp *.py kzisiadis@twmb-kaenguru.nat.uni-magdeburg.de:~/code/vicsek-simulation/
            ;;

        "ko" | "koala")
            scp *.py kzisiadis@twmb-koala.nat.uni-magdeburg.de:~/code/vicsek-simulation/
            ;;
        
        "s" | "spatz")
            scp *.py kzisiadis@twmb-spatz.nat.uni-magdeburg.de:~/code/vicsek-simulation/
            ;;

        "*" | "all")
            scp *.py kzisiadis@twmb-kaenguru.nat.uni-magdeburg.de:~/code/vicsek-simulation/
            scp *.py kzisiadis@twmb-koala.nat.uni-magdeburg.de:~/code/vicsek-simulation/
            scp *.py kzisiadis@twmb-spatz.nat.uni-magdeburg.de:~/code/vicsek-simulation/
            ;;

        *)
            echo "Specified target unknown"
            selectUplaodTarget
            ;;
    esac
}

selectUplaodTarget