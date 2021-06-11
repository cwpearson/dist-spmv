#pragma once

template<typename ForwardIt>
void shift_left(ForwardIt first, ForwardIt last, size_t n) {
    while(first != last) {
        *(first-n) = *first;
        ++first;
    }
}