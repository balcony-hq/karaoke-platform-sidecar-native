/*
 * Append a native model pack to an executable without changing its loader
 * format.  ELF and PE both permit trailing bytes, so the resulting file is a
 * single self-contained sidecar while retaining the normal executable image.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

#define TRAILER_BYTES 16u
static const unsigned char TRAILER_MAGIC[8] = {
    'V', 'S', 'C', 'E', 'M', 'B', '0', '1'
};

static int copy_stream(FILE *destination, FILE *source) {
    const size_t buffer_bytes = 1024u * 1024u;
    unsigned char *buffer = (unsigned char *)malloc(buffer_bytes);
    if (buffer == NULL) return 0;

    int result = 1;
    for (;;) {
        size_t count = fread(buffer, 1, buffer_bytes, source);
        if (count > 0 && fwrite(buffer, 1, count, destination) != count) {
            result = 0;
            break;
        }
        if (count < buffer_bytes) {
            if (ferror(source)) result = 0;
            break;
        }
    }
    free(buffer);
    return result;
}

static int file_length(FILE *file, uint64_t *length) {
    if (fseek(file, 0, SEEK_END) != 0) return 0;
    long position = ftell(file);
    if (position < 0 || fseek(file, 0, SEEK_SET) != 0) return 0;
    *length = (uint64_t)position;
    return 1;
}

static void write_u64_le(unsigned char *destination, uint64_t value) {
    for (unsigned int i = 0; i < 8; i++) {
        destination[i] = (unsigned char)(value & 0xffu);
        value >>= 8;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s EXECUTABLE MODEL OUTPUT\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], argv[3]) == 0 || strcmp(argv[2], argv[3]) == 0) {
        fprintf(stderr, "input and output paths must be different\n");
        return 2;
    }

    FILE *executable = fopen(argv[1], "rb");
    FILE *model = fopen(argv[2], "rb");
    FILE *output = NULL;
    int result = 1;
    if (executable == NULL || model == NULL) {
        fprintf(stderr, "cannot open executable or model input\n");
        goto done;
    }

    uint64_t model_bytes = 0;
    if (!file_length(model, &model_bytes) || model_bytes == 0 || model_bytes > UINT64_MAX - TRAILER_BYTES) {
        fprintf(stderr, "invalid model size\n");
        goto done;
    }
    output = fopen(argv[3], "wb");
    if (output == NULL || !copy_stream(output, executable) || !copy_stream(output, model)) {
        fprintf(stderr, "cannot write embedded sidecar\n");
        goto done;
    }

    unsigned char trailer[TRAILER_BYTES];
    memcpy(trailer, TRAILER_MAGIC, sizeof(TRAILER_MAGIC));
    write_u64_le(trailer + sizeof(TRAILER_MAGIC), model_bytes);
    if (fwrite(trailer, 1, sizeof(trailer), output) != sizeof(trailer)) {
        fprintf(stderr, "cannot write embedded sidecar trailer\n");
        goto done;
    }
    if (fflush(output) != 0) {
        fprintf(stderr, "cannot flush embedded sidecar\n");
        goto done;
    }
#if !defined(_WIN32)
    {
        struct stat details;
        if (stat(argv[1], &details) == 0 && chmod(argv[3], details.st_mode & 0777u) != 0) {
            fprintf(stderr, "cannot preserve executable permissions\n");
            goto done;
        }
    }
#endif
    result = 0;

done:
    if (output != NULL && fclose(output) != 0) result = 1;
    if (model != NULL) fclose(model);
    if (executable != NULL) fclose(executable);
    if (result != 0) remove(argv[3]);
    return result;
}
