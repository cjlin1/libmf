#include <iostream>
#include <string>
#include <cstring>
#include "mf.h"

namespace
{

enum FileType {kData, kModel};

struct ViewOption
{
    std::string path;
    FileType file_type;
};

void view_help()
{
    printf(
    "usage: libmf view {-d|-m} file\n"
    "-d: view a data file\n"
    "-m: view a model file\n"
    );
}

std::shared_ptr<ViewOption> parse_view_option(
        int const argc, char const * const * const argv)
{
    if(argc != 2)
    {
        view_help();
        return std::shared_ptr<ViewOption>(nullptr);
    }
    std::shared_ptr<ViewOption> option(new ViewOption);

    if(!strcmp(argv[0], "-d"))
    {
        option->file_type = kData;
    }
    else if(!strcmp(argv[0], "-m"))
    {
        option->file_type = kModel;
    }
    else
    {
        fprintf(stderr, "Error: invalid option %s\n", argv[0]);
        return std::shared_ptr<ViewOption>(nullptr);
    }

    option->path = std::string(argv[1]);
    return option;
}

bool view_data(std::string const &path)
{
    std::shared_ptr<Matrix> M = read_matrix_meta(path);
    if(!M)
        return false;
    printf("number of users   = %d\n", M->nr_users);
    printf("number of items   = %d\n", M->nr_items);
    printf("number of ratings = %ld\n", M->nr_ratings);
    printf("average           = %f\n", M->avg);
    return true;
}

bool view_model(std::string const &path)
{
    std::shared_ptr<Model> model = read_model_meta(path);
    if(!model)
        return false;
    printf("number of users = %d\n", model->nr_users);
    printf("number of items = %d\n", model->nr_items);
    printf("dimensions      = %d\n", model->param.dim);
    printf("lambda p        = %f\n", model->param.lp);
    printf("lambda q        = %f\n", model->param.lq);
    printf("lambda ub       = %f\n", model->param.lub);
    printf("lambda ib       = %f\n", model->param.lib);
    printf("gamma           = %f\n", model->param.gamma);
    printf("average         = %f\n", model->avg);
    return true;
}

} //namespace

int view(int const argc, char const * const * const argv)
{
    if(argc == 0)
    {
        view_help();
        return EXIT_FAILURE;
    }

    std::shared_ptr<ViewOption> option = parse_view_option(argc, argv);
    if(!option)
        return EXIT_FAILURE;

    if(option->file_type == kData)
    {
        if(!view_data(option->path))
            return EXIT_FAILURE;
    }
    else
    {
        if(!view_model(option->path))
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
