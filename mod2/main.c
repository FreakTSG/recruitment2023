#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define FIFO_NAME "my_pipe"

int main() {
    char buffer[BUFSIZ];
    int fd;
    ssize_t numBytes;

    // Create a named pipe
    if (mkfifo(FIFO_NAME, 0666) == -1) {
        perror("mkfifo");
        exit(EXIT_FAILURE);
    }

    printf("Waiting for Python to connect...\n");

    // Open the named pipe for reading
    fd = open(FIFO_NAME, O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    printf("Python connected. Waiting for data...\n");

    while (1) {
        numBytes = read(fd, buffer, sizeof(buffer));
        if (numBytes > 0) {
            buffer[numBytes] = '\0';
            printf("Received prediction: %s\n", buffer);
        }
    }

    close(fd);

    return 0;
}
