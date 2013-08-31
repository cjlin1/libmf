#include "mf.h"

void exit_main() {
    printf(
        "usage: mf COMMAND [ARGS]\n"
        "\n"
        "Commands include:\n"
        "    convert    Convert a text file to a binary file\n"
        "    train      Train a model from training data\n"
        "    predict    Predict a test data from a model\n"
        "    view       View model and data info\n"
        "\n"
        "See 'mf COMMAND' for more information on a specific command.\n"
    ); 
    exit(1);
}

int main(int argc, char **argv) {
    if(argc<2) exit_main();

    if(!strcmp(argv[1],"convert")) convert(argc,argv);
    else if(!strcmp(argv[1],"train")) train(argc,argv);
    else if(!strcmp(argv[1],"predict")) predict(argc,argv);
    else if(!strcmp(argv[1],"view")) view(argc,argv);
    else printf("Invalid command: %s\n",argv[1]);

    return 0;
}
