
/*
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <string>

#include "Comparator.h"
#include "VideoFolder.h"

using namespace std;

int main(int argc, char *argv[]) {

	if (argc != 3) {
		cout << "first :" << argv[0] << endl;
		cout << "second :" << argv[1] << endl;
		cout << "third :" << argv[2] << endl;
		cerr << "Usage : comparator.exe path\\to\\video\\ path\\to\\binaryFolder\\\n";
		return 0;
	}

	const string videoPath = argv[1];
	const string binaryPath = argv[2];

	bool error = false;
	VideoFolder videoFolder(videoPath, binaryPath);
	try {
		videoFolder.readTemporalFile();
		videoFolder.setExtension();
	} catch (const string &s) {
		error = true;
		cout << "An exception has occured : " << s << "\n";
	}

	if (!error) {
		Comparator comparator(videoFolder);

		comparator.compare();
		comparator.save();
	}

	return 0;
}
