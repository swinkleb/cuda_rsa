cuda_rsa
========

This implementaion has been successfully compiled on 255x10.

To ssh into 255x10 first ssh into unix{1,2,3,4}.

Run the setup.sh script to setup your path to include the necessary items in your path. This will only set these variables for this session.  If you wish to reset these variables during this session run the 'bash' command.

If the gmp library still cannot be found at runtime then issue the following command:

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/clupo/gmp/lib/
