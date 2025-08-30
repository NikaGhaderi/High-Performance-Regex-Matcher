#include "common.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define MAX_LINES 20000000
#define THREADS_PER_BLOCK 256
#define MAX_STATES 2048  // Increased to handle complex patterns
#define MAX_LINE_LENGTH 2048
#define MAX_POSTFIX_LENGTH 8000

__device__ __inline__ int device_isspace(char c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
}

__device__ __inline__ char*
re2post(char *re, int regex_length, char *buf)
{
    int nalt, natom;
    char *dst;
    struct {
        int nalt;
        int natom;
    } paren[100], *p;

    p = paren;
    dst = buf;
    nalt = 0;
    natom = 0;
    if (regex_length >= MAX_POSTFIX_LENGTH || regex_length <= 0) {
        printf("Invalid regex_length in re2post: %d\n", regex_length);
        return NULL;
    }
    for (; *re; re++) {
        switch (*re) {
            case '(':
                if (natom > 1) {
                    --natom;
                    *dst++ = '.';
                }
                if (p >= paren + 100)
                    return NULL;
                p->nalt = nalt;
                p->natom = natom;
                p++;
                nalt = 0;
                natom = 0;
                break;
            case '|':
                if (natom == 0)
                    return NULL;
                while (--natom > 0)
                    *dst++ = '.';
                nalt++;
                break;
            case ')':
                if (p == paren)
                    return NULL;
                if (natom == 0)
                    return NULL;
                while (--natom > 0)
                    *dst++ = '.';
                for (; nalt > 0; nalt--)
                    *dst++ = '|';
                --p;
                nalt = p->nalt;
                natom = p->natom;
                natom++;
                break;
            case '*':
            case '+':
            case '?':
                if (natom == 0)
                    return NULL;
                *dst++ = *re;
                break;
            default:
                if (natom > 1) {
                    --natom;
                    *dst++ = '.';
                }
                *dst++ = *re;
                natom++;
                break;
        }
    }
    if (p != paren)
        return NULL;
    while (--natom > 0)
        *dst++ = '.';
    for (; nalt > 0; nalt--)
        *dst++ = '|';
    *dst = 0;
    return buf;
}

enum
{
    Match = 256,
    Split = 257
};

typedef struct State State;
struct State
{
    int c;
    State *out;
    State *out1;
};

__constant__ State matchstate = { Match };

typedef struct Frag Frag;
struct Frag
{
    State *start;
    State **out;
    int out_count;
};

__device__ __inline__ Frag
frag(State *start, State **out, int out_count)
{
    Frag n = { start, out, out_count };
    return n;
}

__device__ __inline__ State*
state(int c, State *out, State *out1, State *states, int *nstate)
{
    if (*nstate >= MAX_STATES) {
        printf("State limit exceeded: %d\n", *nstate);
        return NULL;
    }
    State *s = &states[*nstate];
    (*nstate)++;
    s->c = c;
    s->out = out;
    s->out1 = out1;
    return s;
}

__device__ __inline__ void
patch(State **out, int out_count, State *s)
{
    for(int i = 0; i < out_count; i++)
        out[i] = s;
}

__device__ __inline__ State**
append(State **l1, int count1, State **l2, int count2, State **out, int *out_count)
{
    *out_count = count1 + count2;
    for (int i = 0; i < count1; i++) out[i] = l1[i];
    for (int i = 0; i < count2; i++) out[i + count1] = l2[i];
    return out;
}

__device__ __inline__ State*
post2nfa(char *postfix, State *states, int *nstate, State **out_buffer)
{
    char *p;
    Frag stack[1000], *stackp, e1, e2, e;
    State *s;
    int out_index = 0;

    if (postfix == NULL) {
        printf("Null postfix in post2nfa\n");
        return NULL;
    }

#define push(s) *stackp++ = s
#define pop() *--stackp

    stackp = stack;
    for (p = postfix; *p; p++) {
        switch (*p) {
            default:
                s = state(*p, NULL, NULL, states, nstate);
                if (!s) return NULL;
                push(frag(s, &s->out, 1));
                break;
            case '.':
                e2 = pop();
                e1 = pop();
                patch(e1.out, e1.out_count, e2.start);
                push(frag(e1.start, e2.out, e2.out_count));
                break;
            case '|':
                e2 = pop();
                e1 = pop();
                s = state(Split, e1.start, e2.start, states, nstate);
                if (!s) return NULL;
                int out_count;
                push(frag(s, append(e1.out, e1.out_count, e2.out, e2.out_count, &out_buffer[out_index], &out_count), out_count));
                out_index += out_count;
                break;
            case '?':
                e = pop();
                s = state(Split, e.start, NULL, states, nstate);
                if (!s) return NULL;
                int out_count_q;
                push(frag(s, append(e.out, e.out_count, &s->out1, 1, &out_buffer[out_index], &out_count_q), out_count_q));
                out_index += out_count_q;
                break;
            case '*':
                e = pop();
                s = state(Split, e.start, NULL, states, nstate);
                if (!s) return NULL;
                patch(e.out, e.out_count, s);
                push(frag(s, &s->out1, 1));
                break;
            case '+':
                e = pop();
                s = state(Split, e.start, NULL, states, nstate);
                if (!s) return NULL;
                patch(e.out, e.out_count, s);
                push(frag(e.start, &s->out1, 1));
                break;
        }
    }

    e = pop();
    if (stackp != stack) {
        printf("Stack imbalance in post2nfa\n");
        return NULL;
    }

    patch(e.out, e.out_count, &matchstate);
    return e.start;

#undef pop
#undef push
}

typedef struct List List;
struct List
{
    State *s[MAX_STATES];
    int n;
};

__device__ __inline__ void
custom_addstate(List *l, State *s)
{
    List state_stack;
    state_stack.n = 0;
    #define push_to_stack(ss) state_stack.s[state_stack.n++] = ss
    #define pop_stack() state_stack.s[--state_stack.n]
    #define is_stack_empty() (state_stack.n == 0)

    push_to_stack(s);
    while(!is_stack_empty()) {
        s = pop_stack();
        if(s == NULL)
            break;
        if(s->c == Split){
            push_to_stack(s->out);
            push_to_stack(s->out1);
        } else {
            if (l->n >= MAX_STATES) {
                printf("List limit exceeded in custom_addstate\n");
                l->n = 0;
                return;
            }
            l->s[l->n++] = s;
        }
    }
    #undef push_to_stack
    #undef pop_stack
    #undef is_stack_empty
}

__device__ __inline__ List*
startlist(State *start, List *l)
{
    l->n = 0;
    custom_addstate(l, start);
    return l;
}

__device__ __inline__ int
ismatch(List *l)
{
    for(int i = 0; i < l->n; i++)
        if(l->s[i]->c == Match)
            return 1;
    return 0;
}

__device__ __inline__ void
step(List *clist, int c, List *nlist, bool case_insensitive)
{
    nlist->n = 0;
    for(int i = 0; i < clist->n; i++){
        State *s = clist->s[i];
        if(s->c == c || (case_insensitive && (
            (s->c >= 'A' && s->c <= 'Z' && s->c + 32 == c) ||
            (s->c >= 'a' && s->c <= 'z' && s->c - 32 == c))))
            custom_addstate(nlist, s->out);
    }
}

__device__ __inline__ int
match(State *start, char *s, List *l1, List *l2, bool case_insensitive)
{
    List *clist, *nlist, *t;
    clist = startlist(start, l1);
    nlist = l2;
    for(; *s; s++){
        int c = *s & 0xFF;
        step(clist, c, nlist, case_insensitive);
        t = clist; clist = nlist; nlist = t;
        if (ismatch(clist)) {
            return 1;
        }
    }
    return ismatch(clist);
}

__global__ void match_kernel(char* regex, int regex_length, char* search_string, int search_string_length,
                            int *linesizes, int number_of_lines, bool case_insensitive) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_of_lines)
        return;

    __shared__ char post_buf[MAX_POSTFIX_LENGTH];
    __shared__ State states[MAX_STATES];
    __shared__ State *out_buffer[2 * MAX_STATES];
    __shared__ int nstate;
    __shared__ State *start;

    if (threadIdx.x == 0) {
        nstate = 0;
        char *post = re2post(regex, regex_length, post_buf);
        if (post == NULL) {
            printf("re2post failed for regex_length=%d\n", regex_length);
            return;
        }
        start = post2nfa(post, states, &nstate, out_buffer);
        if (start == NULL) {
            printf("post2nfa failed, nstate=%d\n", nstate);
            return;
        }
        printf("NFA built with %d states\n", nstate);
    }

    __syncthreads();
    if (start == NULL) return;

    int offset = (index == 0) ? 0 : linesizes[index - 1];
    int end_offset = linesizes[index];

    if (end_offset < offset || end_offset > search_string_length) {
        printf("Invalid line offsets: index=%d, offset=%d, end_offset=%d, search_string_length=%d\n",
               index, offset, end_offset, search_string_length);
        return;
    }

    if (end_offset - offset >= MAX_LINE_LENGTH) {
        printf("Line too long at index=%d, length=%d\n", index, end_offset - offset);
        return;
    }

    char my_search_string[MAX_LINE_LENGTH];
    int i = 0;
    for (int k = offset; k < end_offset && i < MAX_LINE_LENGTH - 1; k++) {
        my_search_string[i] = search_string[k];
        i++;
    }
    my_search_string[i] = '\0';

    List l1, l2;
    if (match(start, my_search_string, &l1, &l2, case_insensitive)) {
        char *end = my_search_string + i - 1;
        while (end >= my_search_string && device_isspace(*end)) *end-- = '\0';
        printf("%s\n", my_search_string);
    }
}

void regexMatchCuda(char* regex, int regex_length, char* search_string, int search_string_length, int *linesizes, int number_of_lines, bool case_insensitive) {
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocks = (number_of_lines + threadsPerBlock - 1) / threadsPerBlock;

    char* device_regex;
    char* device_search_string;
    int* device_linesizes;

    printf("regex_length=%d, search_string_length=%d, number_of_lines=%d\n", 
           regex_length, search_string_length, number_of_lines);
    if (regex_length <= 0 || search_string_length <= 0 || number_of_lines <= 0) {
        fprintf(stderr, "Invalid parameters: regex_length=%d, search_string_length=%d, number_of_lines=%d\n",
                regex_length, search_string_length, number_of_lines);
        return;
    }

    double startTime = get_current_time();
    CUDA_CHECK(cudaMalloc(&device_regex, sizeof(char)*(regex_length+1)));
    CUDA_CHECK(cudaMalloc(&device_search_string, sizeof(char)*(search_string_length+1)));
    CUDA_CHECK(cudaMalloc(&device_linesizes, sizeof(int)*number_of_lines));

    CUDA_CHECK(cudaMemcpy(device_regex, regex, sizeof(char)*(regex_length+1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_search_string, search_string, sizeof(char)*(search_string_length+1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_linesizes, linesizes, sizeof(int)*number_of_lines, cudaMemcpyHostToDevice));

    double kernelStartTime = get_current_time();
    match_kernel<<<blocks, threadsPerBlock>>>(device_regex, regex_length, device_search_string, 
                                             search_string_length, device_linesizes, number_of_lines, case_insensitive);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double kernelEndTime = get_current_time();

    double endTime = get_current_time();

    printf("\nOverall time:                     %.3f s\n", endTime - startTime);
    printf("Kernel time:                      %.3f s\n", kernelEndTime - kernelStartTime);
    printf("Non-kernel (Memcpy+Malloc) time:  %.3f s\n", (kernelStartTime - startTime) + (endTime - kernelEndTime));

    CUDA_CHECK(cudaFree(device_regex));
    CUDA_CHECK(cudaFree(device_search_string));
    CUDA_CHECK(cudaFree(device_linesizes));
}
