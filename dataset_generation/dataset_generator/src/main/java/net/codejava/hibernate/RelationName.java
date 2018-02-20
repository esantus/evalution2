package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="relationName")
public class RelationName {
	private Long relationName_id;
	private String relationName_value;
	
	
	public RelationName() {
		super();
		// TODO Auto-generated constructor stub
	}

	public RelationName(Long relationName_id, String relationName_value) {
		super();
		this.relationName_id = relationName_id;
		this.relationName_value = relationName_value;
	}

	@Id
	@Column(name = "relationName_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getRelationName_id() {
		return relationName_id;
	}

	public void setRelationName_id(Long relationName_id) {
		this.relationName_id = relationName_id;
	}

	public String getRelationName_value() {
		return relationName_value;
	}

	public void setRelationName_value(String relationName_value) {
		this.relationName_value = relationName_value;
	}

	@Override
	public String toString() {
		return "RelationName [relationName_id=" + relationName_id + ", relationName_value=" + relationName_value + "]";
	}
	
	
}
