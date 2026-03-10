#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
