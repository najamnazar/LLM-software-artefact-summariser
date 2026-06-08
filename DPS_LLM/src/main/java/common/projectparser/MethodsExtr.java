package common.projectparser;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import common.utils.Utils;

import java.util.ArrayList;
import java.util.HashMap;

public class MethodsExtr extends VoidVisitorAdapter<Void> {
    private final ArrayList<HashMap<String, Object>> methodsInfoHashArray;

    public MethodsExtr() {
        this.methodsInfoHashArray = new ArrayList<>();
    }

    public ArrayList<HashMap<String, Object>> getMethodInfo(CompilationUnit compilationUnit) {
        compilationUnit.accept(new VoidVisitorAdapter<Void>() {
            @Override
            public void visit(MethodDeclaration md, Void arg) {
                HashMap<String, Object> methodInfo = new HashMap<>();

                // MN
                methodInfo.put("METHODNAME", md.getName().asString());

                // MRT
                methodInfo.put("METHODRETURNTYPE", md.getType().asString());

                // MMT
                methodInfo.put("METHODMODIFIERTYPE", Utils.nodeListToArrayList(md.getModifiers()));
                methodInfo.put("METHODOVERRIDE", Utils.nodeListToArrayList(md.getAnnotations()));
                methodInfo.put("METHODPARAMETER", Utils.getParameters(md.getParameters()));
                methodInfo.put("INCOMINGMETHOD", new ArrayList<>());
                methodInfo.put("OUTGOINGMETHOD", new ArrayList<>());

                getMethodBodyInfo(md, methodInfo, compilationUnit);
                methodsInfoHashArray.add(methodInfo);
            }
        }, null);

        return methodsInfoHashArray;
    }

    private void getMethodBodyInfo(MethodDeclaration md, HashMap<String, Object> methodInfo,
            CompilationUnit compilationUnit) {

        int numberOfMethodVariablesOrAttributes = 0;
        int numberOfMethodLines = 0;
        ArrayList<String> methodBodyLineType = new ArrayList<>();
        if (md.getBody().isPresent()) {
            for (Statement statement : md.getBody().get().getStatements()) {

                // MBLT - To be changed
                {
                    if (statement.isExpressionStmt()) {
                        methodBodyLineType.add("EXPRESSION");
                    } else if (statement.isIfStmt()) {
                        methodBodyLineType.add("IF");
                    } else if (statement.isWhileStmt() ||
                            statement.isForEachStmt() ||
                            statement.isDoStmt() ||
                            statement.isForStmt()) {
                        methodBodyLineType.add("LOOP");
                    } else if (statement.isReturnStmt()) {
                        methodBodyLineType.add("RETURN");
                    } else if (statement.isSwitchStmt()) {
                        methodBodyLineType.add("SWITCH");
                    } else {
                        methodBodyLineType.add("OTHER");
                    }
                }
                if (statement.isExpressionStmt() && statement.asExpressionStmt().getExpression()
                        .isVariableDeclarationExpr()) {
                    numberOfMethodVariablesOrAttributes += statement.asExpressionStmt().getExpression()
                            .asVariableDeclarationExpr().getVariables().size();
                }
            }
            numberOfMethodLines = md.getBody().get().getStatements().size();
        }

        // MBLT - To be changed
        methodInfo.put("METHODBODYLINETYPE", methodBodyLineType.toArray());

        // NOMV/NOMA
        methodInfo.put("NUMBEROFMETHODVARIABLES", numberOfMethodVariablesOrAttributes);

        // NOML
        methodInfo.put("NUMBEROFMETHODLINES", numberOfMethodLines);

        // NIM
        methodInfo.put("NUMBEROFINCOMINGMETHODS", 0);

        // NOM
        methodInfo.put("NUMBEROFOUTGOINGMETHODS", md.findAll(MethodCallExpr.class).size());
    }
}
