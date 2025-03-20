/** @file
* String Utils
*/
#pragma once

#define COMPLIER_SUPPORT_STD_FROM_CHARS 0 ///< Compliler supports std::from_chars

#include "burt/mathroutines/include/SimpleMathRoutines.h"

#include <string>
#include <string_view>
#include <sstream>
#include <vector>

#if COMPLIER_SUPPORT_STD_FROM_CHARS
    #include <charconv>
#endif

#include <stddef.h>
#include <assert.h>

namespace burt
{
    namespace string_utils
    {
        /** Convert arg1 and arg2 to string and concatenate these strings
        * @param arg1 first object
        * @param arg2 second object
        * @return concatenation of two objects
        */
        template <class T1, class T2>
        std::string concat(const T1& arg1, const T2& arg2)
        {
            std::ostringstream s;
            s << arg1;
            s << arg2;
            return s.str();
        }

        /** Convert arg1 arg2 arg3 to string and concatenate these strings
        * @param arg1 first object
        * @param arg2 second object
        * @param arg3 third object
        * @return concatenation of three objects
        */
        template <class T1, class T2, class T3>
        std::string concat(const T1& arg1, const T2& arg2, const T3& arg3)
        {
            std::ostringstream s;
            s << arg1 << arg2 << arg3;
            return s.str();
        }

        /** Parse text string to an element
        * @param element object in which deserialization will be performed
        * @param str string with value, which contains only the 'element' value
        * @return true if convert operation completed successfully
        * @tparam T result type
        */
        template<class T>
        bool fromString(T& restrict_ext element, const char* strBegin, const char* strEnd)
        {
#if COMPLIER_SUPPORT_STD_FROM_CHARS
            // Ignore first plus sign
            if (*strBegin == '+')
                strBegin++;

            std::from_chars_result parseResult = std::from_chars(strBegin, strEnd, element);
            if (parseResult.ec == std::errc())
                return true;
            else
                return false;
#else
            std::string str(strBegin, strEnd);
            std::istringstream s(str);
            s >> element;
            if (!s.eof())
                return false;
            return true;
#endif
        }

        template<class T>
        /**
         * Converts a substring of characters to an unsigned integer.
         *
         * This method converts a sequence of characters defined by the range [strBegin, strEnd) to an unsigned integer
         * and stores the result in the provided element. The method handles optional leading '+' sign.
         *
         * @param element A reference to the variable where the converted unsigned integer will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        inline bool fromStringUnsignedInteger(T& restrict_ext element, const char* strBegin, const char* strEnd)
        {

            // Check used encodings
            static_assert('0' + 1 == '1');
            static_assert('0' + 2 == '2');
            static_assert('0' + 3 == '3');
            static_assert('0' + 4 == '4');
            static_assert('0' + 5 == '5');
            static_assert('0' + 6 == '6');
            static_assert('0' + 7 == '7');
            static_assert('0' + 8 == '8');
            static_assert('0' + 9 == '9');

            // Check that string is not empty
            if (strBegin == strEnd)
                return false;

            if (*strBegin == '+')
            {
                // skip plus sign
                strBegin++;

                // parse first symbol
                if (*strBegin < '0' || *strBegin > '9')
                    return false;
                int symbol = (*strBegin) - '0';
                element = T(+symbol);

                // Skip one parsed digit
                strBegin++;

                // Next iterate through the whole the last elements [integer uses base 10]
                for (; strBegin != strEnd; strBegin++)
                {
                    if (*strBegin < '0' || *strBegin > '9')
                        return false;
                    int symbol = (*strBegin) - '0';
                    element = element * T(10) + T(symbol);
                }
            }
            else
            {
                if (*strBegin < '0' || *strBegin > '9')
                    return false;
                int symbol = (*strBegin) - '0';
                element = T(+symbol);

                //Skip one parsed digit
                strBegin++;

                // Next iterate through the whole the last elements [integer uses base 10]
                for (; strBegin != strEnd; strBegin++)
                {
                    if (*strBegin < '0' || *strBegin > '9')
                        return false;
                    int symbol = (*strBegin) - '0';
                    element = element * T(10) + T(symbol);
                }
            }

            return true;
        }

        /**
         * Converts a substring of characters to a signed integer.
         *
         * This method converts a sequence of characters defined by the range [strBegin, strEnd) to a signed integer
         * and stores the result in the provided element. The method handles optional leading '+' and '-' signs.
         *
         * @param element A reference to the variable where the converted signed integer will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<class T>
        inline bool fromStringSignedInteger(T& restrict_ext element, const char* strBegin, const char* strEnd)
        {
            // Check used encodings
            static_assert('0' + 1 == '1');
            static_assert('0' + 2 == '2');
            static_assert('0' + 3 == '3');
            static_assert('0' + 4 == '4');
            static_assert('0' + 5 == '5');
            static_assert('0' + 6 == '6');
            static_assert('0' + 7 == '7');
            static_assert('0' + 8 == '8');
            static_assert('0' + 9 == '9');

            if (strBegin == strEnd)
                return false;

            if (*strBegin == '+')
            {
                // Skip plus sign
                strBegin++;

                if (*strBegin < '0' || *strBegin > '9')
                    return false;

                int symbol = (*strBegin) - '0';
                element = T(+ symbol);

                // Skip one parsed digit
                strBegin++;

                // Iterate through the whole elements
                for (; strBegin != strEnd; strBegin++)
                {
                    if (*strBegin < '0' || *strBegin > '9')
                        return false;
                    int symbol = (*strBegin) - '0';
                    element = element * T(10) + T(symbol);
                }
            }
            else if (*strBegin == '-')
            {
                // Skip negative sign
                strBegin++;

                if (*strBegin < '0' || *strBegin > '9')
                    return false;

                int symbol = (*strBegin) - '0';
                element = T(-symbol);

                // Skip one parsed digit
                strBegin++;

                // Iterate through the whole elements
                for (; strBegin != strEnd; strBegin++)
                {
                    if (*strBegin < '0' || *strBegin > '9')
                        return false;
                    int symbol = (*strBegin) - '0';
                    element = element * T(10) - T(symbol);
                }
            }
            else
            {
                if (*strBegin < '0' || *strBegin > '9')
                    return false;

                int symbol = (*strBegin) - '0';
                element = T(+symbol);

                // Skip one parsed digit
                strBegin++;

                // Iterate through the whole elements
                for (; strBegin != strEnd; strBegin++)
                {
                    if (*strBegin < '0' || *strBegin > '9')
                        return false;
                    int symbol = (*strBegin) - '0';
                    element = element * T(10) + T(symbol);
                }
            }

            return true;
        }

        /**
         * Converts a substring of characters to a floating-point number.
         *
         * This method parses a sequence of characters defined by the range [strBegin, strEnd) to a floating-point number
         * and stores the result in the provided element. The method handles optional leading '+' and '-' signs, as well as
         * optional fractional and exponential parts.
         *
         * @param element A reference to the variable where the converted floating-point number will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<class T>
        inline bool fromStringFloat(T& restrict_ext element, const char* strBegin, const char* strEnd)
        {
            // Check used encodings
            static_assert('0' + 1 == '1');
            static_assert('0' + 2 == '2');
            static_assert('0' + 3 == '3');
            static_assert('0' + 4 == '4');
            static_assert('0' + 5 == '5');
            static_assert('0' + 6 == '6');
            static_assert('0' + 7 == '7');
            static_assert('0' + 8 == '8');
            static_assert('0' + 9 == '9');

            if (strBegin == strEnd)
                return false;

            int signIntegerPart = +1;
            int integerPart = 0;
            int fracPart = 0;
            uint32_t fracPartLength = 0;

            int signExpPart = +1;
            int expPart = 0;

            if (*strBegin == '+')
            {
                // signIntegerPart = +1;
                // skip plus sign
                strBegin++;
            }
            else if (*strBegin == '-')
            {
                signIntegerPart = -1;
                // skip minus sign
                strBegin++;
            }

            // Parse integer first part of float/double text representation
            for (; strBegin != strEnd; strBegin++)
            {
                if (*strBegin == '.' || *strBegin == 'e' || *strBegin == 'E')
                {
                    // Move to the next part
                    break;
                }

                if (*strBegin < '0' || *strBegin > '9')
                {
                    return false;
                }

                int symbol = (*strBegin) - '0';
                integerPart = integerPart * 10 + symbol;
            }

            // Parse fractional part
            if (strBegin != strEnd && *strBegin == '.')
            {
                // Skip '.' character
                strBegin++;

                for (; strBegin != strEnd; strBegin++)
                {
                    if (*strBegin == 'e' || *strBegin == 'E')
                    {
                        // Move to the next part
                        break;
                    }

                    if (*strBegin < '0' || *strBegin > '9')
                    {
                        return false;
                    }

                    int symbol = (*strBegin) - '0';
                    fracPart = fracPart * 10 + symbol;
                    fracPartLength++;
                }
            }

            // Parse exponential part
            if (strBegin != strEnd && (*strBegin == 'E' || *strBegin == 'e') )
            {
                // Skip 'E' or 'e' character
                strBegin++;

                if (*strBegin == '+')
                {
                    // signExpPart = +1;
                    strBegin++; // skip plus sign in exp. part
                }
                else if (*strBegin == '-')
                {
                    signExpPart = -1;
                    strBegin++; // skip minus sign in exp. part
                }

                for (; strBegin != strEnd; strBegin++)
                {
                    if (*strBegin < '0' || *strBegin > '9')
                    {
                        return false;
                    }
                    int symbol = (*strBegin) - '0';
                    expPart = expPart * 10 + symbol;
                }
            }

            if (expPart == 0)
            {
                if (fracPartLength == 0)
                {
                    if (signIntegerPart == +1)
                    {
                        element = T(integerPart);
                    }
                    else /*if (signIntegerPart == -1)*/
                    {
                        element = -T(integerPart);
                    }
                }
                else
                {
                    //==============================================================//
                    static constexpr T fracDenominators[] = { T(1)/ T(1e0),
                                                              T(1) / T(1e1),
                                                              T(1) / T(1e2),
                                                              T(1) / T(1e3),
                                                              T(1) / T(1e4),
                                                              T(1) / T(1e5),
                                                              T(1) / T(1e6),
                                                              T(1) / T(1e7),
                                                              T(1) / T(1e8),
                                                              T(1) / T(1e9),
                                                              T(1) / T(1e10),
                                                              T(1) / T(1e11),
                                                              T(1) / T(1e12),
                                                              T(1) / T(1e13),
                                                              T(1) / T(1e14),
                                                              T(1) / T(1e15),
                                                              T(1) / T(1e16)
                                                            };

                    T fracDenominator = T();

                    if (fracPartLength < sizeof(fracDenominators) / sizeof(fracDenominators[0])) {
                        fracDenominator = fracDenominators[fracPartLength];
                    } else {
                        fracDenominator = T(1) / powerNatural(T(10), fracPartLength);
                    }
                    //==============================================================//

                    if (signIntegerPart == +1)
                    {
                        element = (T(integerPart) + T(fracPart) * fracDenominator);
                    }
                    else /* if (signIntegerPart == -1) */
                    {
                        element = -(T(integerPart) + T(fracPart) * fracDenominator);
                    }
                }
            }
            else
            {
                //========================================================================================================//
                T fracDenominator = T(1) / powerNatural(T(10), fracPartLength);
                T expPartValue = powerNatural(T(10), uint32_t(expPart));
                //========================================================================================================//

                {
                    if (signIntegerPart == +1 && signExpPart == +1)
                    {
                        element = (T(integerPart) + T(fracPart) * fracDenominator) * expPartValue;
                    }
                    else if (signIntegerPart == +1 && signExpPart == -1)
                    {
                        element = (T(integerPart) + T(fracPart) * fracDenominator) / expPartValue;
                    }
                    else if (signIntegerPart == -1 && signExpPart == +1)
                    {
                        element = -(T(integerPart) + T(fracPart) * fracDenominator) * expPartValue;
                    }
                    else if (signIntegerPart == -1 && signExpPart == -1)
                    {
                        element = -(T(integerPart) + T(fracPart) * fracDenominator) / expPartValue;
                    }
                    else
                    {
                        assert(false);
                    }
                }
            }

            return true;
        }

        /**
         * Converts a substring of characters to an integer.
         *
         * This method converts a sequence of characters defined by the range [strBegin, strEnd) to an integer
         * and stores the result in the provided element.
         *
         * @param element A reference to the variable where the converted integer will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<>
        inline bool fromString(int& restrict_ext element, const char* strBegin, const char* strEnd) {
            return fromStringSignedInteger(element, strBegin, strEnd);
        }

        /**
         * Converts a substring of characters to a long integer.
         *
         * This method converts a sequence of characters defined by the range [strBegin, strEnd) to a long integer
         * and stores the result in the provided element.
         *
         * @param element A reference to the variable where the converted long integer will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<>
        inline bool fromString(long& restrict_ext element, const char* strBegin, const char* strEnd) {
            return fromStringSignedInteger(element, strBegin, strEnd);
        }

        /**
         * Converts a substring of characters to a signed long long integer.
         *
         * This method converts a sequence of characters defined by the range [strBegin, strEnd) to a signed long long integer
         * and stores the result in the provided element.
         *
         * @param element A reference to the variable where the converted signed long long integer will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<>
        inline bool fromString(long long& restrict_ext element, const char* strBegin, const char* strEnd) {
            return fromStringSignedInteger(element, strBegin, strEnd);
        }

        /**
         * Converts a substring of characters to an unsigned integer.
         *
         * This method delegates the conversion of a sequence of characters defined by the range [strBegin, strEnd)
         * to an unsigned integer and stores the result in the provided element.
         *
         * @param element A reference to the variable where the converted unsigned integer will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<>
        inline bool fromString(unsigned int& restrict_ext element, const char* strBegin, const char* strEnd) {
            return fromStringUnsignedInteger(element, strBegin, strEnd);
        }

        template<>
        inline bool fromString(unsigned long& restrict_ext element, const char* strBegin, const char* strEnd) {
            return fromStringUnsignedInteger(element, strBegin, strEnd);
        }

        template<>
        inline bool fromString(unsigned long long& restrict_ext element, const char* strBegin, const char* strEnd) {
            return fromStringUnsignedInteger(element, strBegin, strEnd);
        }

        /**
         * Converts a substring of characters to a floating-point number.
         *
         * This method converts a sequence of characters defined by the range [strBegin, strEnd) to a floating-point number
         * and stores the result in the provided element.
         *
         * @param element A reference to the variable where the converted floating-point number will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<>
        inline bool fromString(float& restrict_ext element, const char* strBegin, const char* strEnd) {
            return fromStringFloat(element, strBegin, strEnd);
        }

        /**
         * Converts a substring of characters to a double precision floating-point number.
         *
         * This method converts a sequence of characters defined by the range [strBegin, strEnd) to a double
         * and stores the result in the provided element.
         *
         * @param element A reference to the variable where the converted double will be stored.
         * @param strBegin A pointer to the beginning of the character sequence to convert.
         * @param strEnd A pointer to the end of the character sequence to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<>
        inline bool fromString(double& restrict_ext element, const char* strBegin, const char* strEnd) {
            return fromStringFloat(element, strBegin, strEnd);
        }

        /**
         * Converts a substring of characters to a specified type.
         *
         * This method converts a given std::string_view to a specified type by internally calling another
         * conversion method that takes pointers to the beginning and end of the character sequence.
         *
         * @param element A reference to the variable where the converted value will be stored.
         * @param strView A view of the string containing the characters to convert.
         * @return Returns true if the conversion was successful, false otherwise.
         */
        template<class T>
        bool fromString(T& restrict_ext element, std::string_view strView)
        {
            return fromString(element, strView.data(), strView.data() + strView.size());
        }

        /** Serialize element to string
        * @param element object which will be serialized
        * @return string with serialized element
        */
        template<class T>
        std::string toString(const T& element)
        {
            std::ostringstream s;
            s << element;
            return s.str();
        }

        inline std::string toString(long element) {
            return std::to_string(element);
        }

        inline std::string toString(long long element) {
            return std::to_string(element);
        }

        inline std::string toString(unsigned long element) {
            return std::to_string(element);
        }

        inline std::string toString(unsigned long long element) {
            return std::to_string(element);
        }

        inline std::string toString(int element) {
            return std::to_string(element);
        }

        inline std::string toString(double element) {
            return std::to_string(element);
        }

        inline std::string toString(float element) {
            return std::to_string(element);
        }

        inline std::string toString(const char* element) {
            return std::string(element);
        }

        /** Serialize element C-array to string
        * @param elementArray start of the array or pointer to the first element
        * @param elementArraySize size of the array
        * @return string with serialized elements
        */
        template<class T>
        std::string toString(const T* elementArray, size_t elementArraySize,
                             const std::string& elementsDelimiter = ",")
        {
            std::ostringstream s;

            if (elementArraySize > 0)
            {
                s << "[";
                for (size_t i = 0; i < elementArraySize - 1; ++i)
                    s << elementArray[i] << elementsDelimiter;
                s << elementArray[elementArraySize - 1];
                s << "]";
            }
            else
            {
                s << "[]";
            }

            return s.str();
        }

        /** Split string "str" into substrings by delimiter "isDelimiter" and "Append" result into "result".
        * @param result result array
        * @param str input string
        * @param isDelimiter function which returns information is the current symbol delimiter or note
        * @return None
        * @remark Important: By default, this function does not return empty substrings
        */
        template <bool returnEmptySubStrings = false, class TStringType, class Func>
        inline void splitToSubstrings(std::vector<std::string_view>& restrict_ext result,
                                      const TStringType& restrict_ext str,
                                      Func isDelimiter)
        {
            size_t start = 0;
            size_t strSize = str.size();
            size_t i = 0;

            for (i = 0; i < strSize; ++i)
            {
                if (isDelimiter(str[i]))
                {
                    size_t count = i - start;

                    if constexpr (returnEmptySubStrings)
                    {
                        // put both empty and not empty lines
                        result.emplace_back(&str[start], count);
                    }
                    else
                    {
                        if (count > 0)
                        {
                            // put only not-empty lines
                            result.emplace_back(&str[start], count);
                        }
                    }

                    start = i + 1;
                }
            }

            // Put the last subline into the stack of results
            size_t count = i - start;
            if (count > 0)
            {
                // put only not-empty last line
                result.emplace_back(&str[start], count);
            }
        }

        /**
         * Splits a given string into a vector of substrings based on a delimiter function.
         *
         * This method processes the input string and splits it into multiple substrings according to the provided delimiter
         * function. The resulting substrings are stored in a vector and returned.
         *
         * @param str The input string to be split into substrings.
         * @param isDelimiter A function or function object that determines if a character is a delimiter.
         * @return A vector containing the split substrings.
         */
        template <bool returnEmptySubStrings = false, class TStringType, class Func>
        inline std::vector<std::string_view> splitToSubstrings(const TStringType& restrict_ext str,
                                                               Func isDelimiter)
        {
            std::vector<std::string_view> res;
            splitToSubstrings<returnEmptySubStrings>(res, str, isDelimiter);
            return res;
        }

        /**
         * Splits a string into a vector of substrings based on a specified delimiter.
         *
         * This method takes a string and splits it into substrings wherever the provided delimiter character is found.
         * The resulting substrings are stored in a vector of string_views.
         *
         * @param str The string to be split into substrings.
         * @param delimiter The character used to split the string. Defaults to ':' if not provided.
         * @return A vector of string_views, each representing a substring of the original string.
        */
        template <bool returnEmptySubStrings = false, class TStringType>
        inline std::vector<std::string_view> splitToSubstrings(const TStringType& restrict_ext str,
                                                               char delimiter = ':')
        {
            auto testFunc = [delimiter](char c)
            {
                return c == delimiter;
            };
            std::vector<std::string_view> res;
            splitToSubstrings<returnEmptySubStrings> (res, str, testFunc);
            return res;
        }
    }
}