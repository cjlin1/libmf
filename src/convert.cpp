#include <string>
#include <iostream>
#include <cstring>
#include "mf.h"

namespace 
{

struct ConvertOption
{
    std::string text_path, binary_path;
};

void convert_help()
{
    printf("usage: libmf convert text_file [binary_file]\n");
}

std::shared_ptr<ConvertOption> parse_convert_option(
        int const argc, char const * const * const argv)
{
    if((argc != 1) && (argc != 2))
    {
        convert_help();
        return std::shared_ptr<ConvertOption>(nullptr);
    }

    std::shared_ptr<ConvertOption> option(new ConvertOption);

    option->text_path = std::string(argv[0]);
    if(argc == 2)
    {
        option->binary_path = std::string(argv[1]);
    }
    else
    {
        const char *p = strrchr(argv[0], '/');
        if(!p)
            p = argv[0];
        else
            p++;
        option->binary_path = std::string(p) + ".bin";
    }
    return option;
}

bool convert(std::string const &text_path, std::string const &binary_path)
{
    FILE *f = fopen(text_path.c_str(), "r");
    if(!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.", text_path.c_str());
        return false;
    }
    Timer timer;
    timer.tic("Converting...");

    Matrix M;
    double sum = 0;
    while(true)
    {
        Node r;
        if(fscanf(f, "%d %d %f\n", &r.uid, &r.iid, &r.rate) == EOF)
            break;
        if(r.uid < 0 || r.iid <0)
        {
            fprintf(stderr, "\nError: User ID and Item ID should not be smaller than zero.\n");
            return false;
        }
        if(r.uid+1 > M.nr_users)
            M.nr_users = r.uid+1;
        if(r.iid+1 > M.nr_items)
            M.nr_items = r.iid+1;
        sum += r.rate;
        M.R.push_back(r);
    }
    M.nr_ratings = (long)(M.R.size());
    M.avg = (float)(sum/M.nr_ratings);

    timer.toc("done.");
    if(!write_matrix(M, binary_path))
        return false;
    fclose(f);
    return true;
}

} // namespace

int convert(int const argc, const char * const * const argv)
{
    std::shared_ptr<ConvertOption> option = parse_convert_option(argc, argv);
    if(!option)
        return EXIT_FAILURE;

    if(!convert(option->text_path, option->binary_path))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
