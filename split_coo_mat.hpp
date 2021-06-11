#pragma once

#include "coo_mat.hpp"
#include "partition.hpp"




/* local matrix has cols renumbered to be 0..N
   into the local dense vector

   remote matrix has cols renumbered for the remote dense vector
*/
struct SplitCooMat {
    int loff; // global row for local matrix 0
    CsrMat<Where::host> local; // local matrix
    CsrMat<Where::host> remote; // remote matrix (with local column indices)
    std::map<int, int> locals;  // get local column from global column
    std::vector<int> globals; // get global column for local column
};

/* Row partition of a matrix into a "local" and "remote" part
   If locally there are rows i...j, then the local part also has columns i...j
   The remote part will have all other columns

   Each rank will also renumber the column indices in the remote part
   This rank will recv the corresponding remote x vector entries, but
   doesn't want to materialize the whole distributed x vector in memory
   so the column indices must be packed into a contiguous range 0...N

   Furthermore, we want all entries from rank 0 to come first, then rank 1, etc.
   This is so we can just get entries from rank 0 and recv them directly into the
   remote x vector at the correct offset

   To do this, relabel the matrix in the following way:
   Get a list of unique required global ids, and then sort them.
   The first will be local 0, then local 1, etc

*/
SplitCooMat split_local_remote(const CsrMat<Where::host> &m, MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // which rows of x are local
    Range localRange = get_partition(m.num_cols(), rank, size);
    int loff = localRange.lb;

    // build two matrices, local gets local non-zeros, remote gets remote non-zeros
    CooMat local(m.num_rows(), m.num_cols());
    CooMat remote(m.num_rows(), m.num_cols());

    std::vector<int> globals; // get global col for local col
    for (int r = 0; r < m.num_rows(); ++r) {
        for (int ci = m.row_ptr(r); ci < m.row_ptr(r+1); ++ci) {
            int c = m.col_ind(ci);
            float v = m.val(ci);
            if (c >= localRange.lb && c < localRange.ub) {
                int lc = c - loff;
                local.push_back(r,lc,v);
            } else {
                // keep the global column for now, it will be renumbered later
                globals.push_back(c);
                remote.push_back(r, c, v);
            }
        }
    }

    // sort required global columns.
    // this will ensure the lowest owning rank comes first, and all are contiguous
    std::sort(globals.begin(), globals.end());
    auto it = std::unique(globals.begin(), globals.end());
    globals.resize(it - globals.begin());

    std::map<int, int> locals; // get local col for global column
    for (size_t lc = 0; lc < globals.size(); ++lc) {
        int gc = globals[lc];
        locals[gc] = lc;
    }

    // relabel remote columns
    for (CooMat::Entry &e : remote) {
        e.j = locals[e.j];
    }

    return SplitCooMat {
        .loff=loff,
        .local=local,
        .remote=remote,
        .locals=locals,
        .globals=globals
    };

}