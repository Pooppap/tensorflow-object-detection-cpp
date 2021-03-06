#include <iostream>
#include <regex>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

tensorflow::Status readLabelsMapFile(const std::string &fileName, map<int, std::string> &labelsMap)
{

    // Read file into a string
    std::ifstream ifs(fileName);
    if (ifs.bad())
        return tensorflow::errors::NotFound("Failed to load labels map at '", fileName, "'");
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string fileString = buffer.str();

    // Search entry patterns of type 'item { ... }' and parse each of them
    std::smatch matcherEntry;
    std::smatch matcherId;
    std::smatch matcherName;
    const std::regex reEntry("item \\{([\\S\\s]*?)\\}");
    const std::regex reId("[0-9]+");
    const std::regex reName("\'.+\'");
    std::string entry;

    auto stringBegin = std::sregex_iterator(fileString.begin(), fileString.end(), reEntry);
    auto stringEnd = std::sregex_iterator();

    int id;
    std::string name;
    for (std::sregex_iterator i = stringBegin; i != stringEnd; i++)
    {
        matcherEntry = *i;
        entry = matcherEntry.str();
        std::regex_search(entry, matcherId, reId);
        if (!matcherId.empty())
            id = std::stoi(matcherId[0].str());
        else
            continue;
        std::regex_search(entry, matcherName, reName);
        if (!matcherName.empty())
            name = matcherName[0].str().substr(1, matcherName[0].str().length() - 2);
        else
            continue;
        labelsMap.insert(pair<int, std::string>(id, name));
    }
    return tensorflow::Status::OK();
}