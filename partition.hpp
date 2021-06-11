#pragma once

#include "csr_mat.hpp"

struct Range {
    int lb;
    int ub;

    int extent() const { return ub-lb; }
};

/* get the ith part of splitting domain in to n pieces
   if not divisible, remainder distributed to lower
*/
Range get_partition(const int domain, const int i, const int n) {
    int div = domain / n;
    int rem = domain % n;

    int lb, ub;

    if (i < rem) {
        lb = i * (div+1);
        ub = lb + (div+1);
    } else {
        lb = rem * (div+1) + (i-rem) * div;
        ub = lb + div;
    }
    return Range{.lb=lb, .ub=ub};
}

// who owns item `i` from `domain` split into `n`
int get_owner(int domain, int i, const int n) {
    int div = domain / n;
    int rem = domain % n;

    // i is in the first, pieces, which are div+1
    if (i < (div + 1) * rem) {
        return i / (div + 1);
    } else {
        i -= (div+1) * rem;
        domain -= (div+1) * rem;
        return rem + i / div;
    }
}

std::vector<CsrMat<Where::host>> part_by_rows(const CsrMat<Where::host> &m, const int parts) {

    std::vector<CsrMat<Where::host>> mats;

    for (int p = 0; p < parts; ++p) {
        Range range = get_partition(m.num_rows(), p, parts);
        std::cerr << "matrix part " << p << " has " << range.ub-range.lb << " rows\n";
        CsrMat<Where::host> part(m);
        part.retain_rows(range.lb, range.ub);
        mats.push_back(part);
    }

    return mats;
}