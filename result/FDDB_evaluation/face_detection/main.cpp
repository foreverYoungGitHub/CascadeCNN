#include <iostream>
#include "fddb_detection.h"
int main() {
    string dataset_path = "/home/xileli/Documents/dateset/FDDB/";

    fddb_detection detection(dataset_path);

    detection.run();
}