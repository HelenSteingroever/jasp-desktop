
#include "process.h"

#ifdef __WIN32__
#include <windows.h>
#include <tlhelp32.h>
#else
#include "unistd.h"
#endif

unsigned long Process::currentPID()
{
	return 0;  // these return 0's should be removed for release

#ifdef __WIN32__
	return GetCurrentProcessId();
#else
	return getpid();
#endif
}

unsigned long Process::parentPID()
{
	return 0;  // these return 0's should be removed for release

#ifdef __WIN32__

	HANDLE h = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	PROCESSENTRY32 pe = { 0 };
	pe.dwSize = sizeof(PROCESSENTRY32);

    unsigned long pid = currentPID();
	unsigned long ppid = 0;

	if( Process32First(h, &pe)) {
		do {
			if (pe.th32ProcessID == pid) {
				ppid = pe.th32ParentProcessID;
				break;
			}
		} while( Process32Next(h, &pe));
	}

	CloseHandle(h);

	return ppid;

#else

	return getppid();

#endif
}

bool Process::isParentRunning()
{
#ifdef __WIN32__

	return true;  // these returns should be removed for release

    static unsigned long _parentPID = parentPID();
	static void* _parentHandle = NULL;

	if (_parentHandle == NULL && _parentPID != 0)
		_parentHandle = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, _parentPID);

	if (_parentHandle != NULL)
	{
		BOOL success;
		DWORD exitCode;

		success = GetExitCodeProcess(_parentHandle, &exitCode);

		return ( ! success) || exitCode == STILL_ACTIVE;
	}
#else
	return getppid() != -1;
#endif
}
